# okc: problem=muse_moe_w4a16
"""Routed NVFP4 (W4A16) MoE experts for one EP8 rank, on grouped Gluon kernels. From 128 tokens (verify) both GEMMs
run on the register path with a per-item token tile: W13 on work items of up to 32 rows (one weight decode for up to
32 token rows, 8/16/32-row tile per item) and W2 on items of up to 16 rows (8/16-row tile per item).

Pipeline (4 launches; 3 at decode 1, where each top-k slot is one item and the routing pass is skipped):
  1. _route (Triton, one CTA): maps global expert ids to this rank's experts, groups the (token, slot) pairs by expert
     (sorted, so pair order within an expert is deterministic) and writes a work list of (expert, first row, end row)
     items of at most BM rows.
  2. _w13_silu (Gluon): for each work item and 64-row N tile, streams the expert's gate and up weight rows once,
     multiplies them with the item's token rows and writes silu(gate) * up (BF16) per pair.
  3. _w2 (Gluon): the same for W2 on the intermediate rows, writing BF16 expert outputs per pair.
  4. SGLang's fused post-expert-norm weighted-sum Triton kernel: RMSNorm with (1 + w), gate weight, sum over slots.

GEMM data path (per CTA: 4 warps, 64 weight rows, K in blocks of 256):
- Weights stay NVFP4. prepare() only permutes the bits inside each 32-bit word of packed E2M1 codes (lossless), so
  that one rotate and one mask place a pair of codes in BF16 bit positions (value * 2^-126); a bf16x2 multiply by
  the E4M3 block scale (times 2^119, exact) completes the decode. The remaining 2^7 and the FP32 global scale are
  applied to the FP32 accumulator.
- Weight words are loaded from global memory straight into the registers of the mma_v2 A operand
  (k_width-2 operand layout; logical K k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 maps to physical K 64 q + 8 w + 4 b3 +
  2 b2 + b0, so each thread's K values are 32 contiguous bytes); the decode happens in those registers, with no
  shared memory for weights.
- Activations are the B operand, loaded once per CTA at every token count. They go through an XOR-swizzled shared
  buffer read by each warp in the B-operand layout (mma_v2), at every token count (VERIFY_SMB = False). With
  VERIFY_SMB, from SMB_MIN_M tokens (verify 16/32) each K block is instead stored as a 128-byte-swizzled [BM, BK]
  tile in the logical K order; on sm_90 (H200) the GEMMs multiply it with warpgroup_mma (wgmma), decoding the
  weights in K chunks of 64 just before each chunk's wgmma (register A operand); elsewhere mma_v2 reads the tile.
- Token tiles: 8 rows for M <= 32 tokens, 16 rows otherwise (shape-based choice).
- W13's register allocation is capped at 96 from SMB_MIN_M tokens so more CTAs fit per SM (H200 measurement).
- At M <= PF_MAX_M tokens (decode 1) each GEMM double-buffers its weight and scale loads in registers: the next K
  block is requested before the current one is decoded, so a CTA has two blocks in flight.
BF16 roundings follow SGLang's path: h rounded to BF16 before SiLU, the SiLU result and the product rounded to BF16,
W2 outputs in BF16.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, warpgroup_mma

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


TOP_K = 8
POST_NORM_EPS = 1e-8
SCALE_BIAS = gl.constexpr(2.0 ** 119)
ACC_BIAS = gl.constexpr(2.0 ** 7)
BM = 16
BN = 64
BK = 256
NUM_WARPS = 4
XS_MIN_M = 1
W13_MAXNREG = {}  # W13 register cap by token count (occupancy, measured on the H200); M >= SMB_MIN_M uses W13_MAXNREG_BIG
W13_MAXNREG_BIG = 96
D1_W13_SPLIT = 2  # K split of the decode-1 W13 (partials reduced by _d1_reduce)
D1_W2_KS = 4  # K split of the decode-1 W2
PF_MAX_M = 1  # weight prefetch (register double buffer) at decode 1 only
IL_MIN_M = 128  # interleaved code+scale blocks from this many tokens ...
IL_M = (8,)  # ... and at these token counts
GU_M = (16, 32)  # token counts whose W13 uses the gate/up-paired copy
SMB_MIN_M = 128  # shared-B path (wgmma on sm_90, mma_v2 elsewhere, which checks its data path) from this many tokens
# The shared-B path is off. From SMB_MIN_M tokens both GEMMs run the register path (shared X
# tile, mma_v2), W13 still capped at 96 registers: on the H200 that is 7-8% faster per
# verify call than shared-B + wgmma, bit-exact. Set True to restore the shared-B verify path.
VERIFY_SMB = False
MIX_MIN_M = 128  # from this many tokens the GEMMs are _w13_mix/_w2_mix (token tile chosen per work item)
# True keeps W2 at verify on the shared-B path (shared-B + wgmma, _w2) and changes only W13. False (used) runs
# W2 on the register path with the per-item tile as well.
W2_VERIFY_C175 = False
# From MIX_MIN_M tokens W13 runs on its own work list of items of up to BM_W13 rows (one weight decode for up
# to 32 token rows; 8/16/32-row tile per item) while W2 keeps the 16-row list; both lists come from one routing pass.
BM_W13 = 32
W13_MAXNREG_MIX = 128  # the 32-row tile needs 128 registers to run without spills


@triton.jit
def _route(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, P, local_start,
           LOCAL: tl.constexpr, BM: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr):
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - local_start
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    tl.store(local_ptr + offs, lid, mask=pm)
    # Pairs grouped by local expert, in pair order within an expert (deterministic); non-local pairs last.
    key = tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = tl.sort(key)
    tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
    starts = tl.cumsum(counts, 0) - counts
    chunks = (counts + BM - 1) // BM
    coff = tl.cumsum(chunks, 0) - chunks
    total = tl.sum(chunks, 0)
    e_ar = tl.arange(0, LOCAL)
    w = tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
    c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
    e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
    e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * BM
    r1 = tl.minimum(r0 + BM, e_start + e_cnt)
    wm = w < total
    tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    tl.store(nwork_ptr, total)


@triton.jit
def _route2(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, work2_ptr, nwork2_ptr, P, local_start,
            LOCAL: tl.constexpr, BM: tl.constexpr, BM2: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr,
            W_MAX2: tl.constexpr):
    """_route, plus a second work list over the same sorted pairs with items of at most BM2 rows (W13's)."""
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - local_start
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    tl.store(local_ptr + offs, lid, mask=pm)
    key = tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = tl.sort(key)
    tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
    starts = tl.cumsum(counts, 0) - counts
    e_ar = tl.arange(0, LOCAL)
    chunks = (counts + BM - 1) // BM
    coff = tl.cumsum(chunks, 0) - chunks
    total = tl.sum(chunks, 0)
    w = tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
    c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
    e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
    e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * BM
    r1 = tl.minimum(r0 + BM, e_start + e_cnt)
    wm = w < total
    tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    tl.store(nwork_ptr, total)
    chunks2 = (counts + BM2 - 1) // BM2
    coff2 = tl.cumsum(chunks2, 0) - chunks2
    total2 = tl.sum(chunks2, 0)
    w2 = tl.arange(0, W_MAX2)
    sel2 = (coff2[None, :] <= w2[:, None]) & ((coff2 + chunks2)[None, :] > w2[:, None])
    owner2 = tl.sum(tl.where(sel2, e_ar[None, :], 0), 1)
    c_first2 = tl.sum(tl.where(sel2, coff2[None, :], 0), 1)
    e_start2 = tl.sum(tl.where(sel2, starts[None, :], 0), 1)
    e_cnt2 = tl.sum(tl.where(sel2, counts[None, :], 0), 1)
    q0 = e_start2 + (w2 - c_first2) * BM2
    q1 = tl.minimum(q0 + BM2, e_start2 + e_cnt2)
    wm2 = w2 < total2
    tl.store(work2_ptr + w2 * 3 + 0, owner2, mask=wm2)
    tl.store(work2_ptr + w2 * 3 + 1, q0, mask=wm2)
    tl.store(work2_ptr + w2 * 3 + 2, q1, mask=wm2)
    tl.store(nwork2_ptr, total2)



@triton.jit
def _route2_mc(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, work2_ptr, nwork2_ptr, P, local_start,
               LOCAL: tl.constexpr, BM: tl.constexpr, BM2: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr,
               W_MAX2: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - local_start
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    if pid < LOCAL:
        start = tl.sum((live & (lid < pid)).to(tl.int32), 0)
        mine = live & (lid == pid)
        rank = tl.cumsum(mine.to(tl.int32), 0) - 1
        tl.store(sorted_ptr + start + rank, offs, mask=mine)
    elif pid == LOCAL:
        dead = pm & (live == 0)
        n_live = tl.sum(live.to(tl.int32), 0)
        rank = tl.cumsum(dead.to(tl.int32), 0) - 1
        tl.store(sorted_ptr + n_live + rank, offs, mask=dead)
    elif pid == LOCAL + 1:
        tl.store(local_ptr + offs, lid, mask=pm)
        counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
        starts = tl.cumsum(counts, 0) - counts
        e_ar = tl.arange(0, LOCAL)
        chunks = (counts + BM - 1) // BM
        coff = tl.cumsum(chunks, 0) - chunks
        total = tl.sum(chunks, 0)
        w = tl.arange(0, W_MAX)
        sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
        owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
        c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
        e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
        e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
        r0 = e_start + (w - c_first) * BM
        r1 = tl.minimum(r0 + BM, e_start + e_cnt)
        wm = w < total
        tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
        tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
        tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
        tl.store(nwork_ptr, total)
    else:
        counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
        starts = tl.cumsum(counts, 0) - counts
        e_ar = tl.arange(0, LOCAL)
        chunks2 = (counts + BM2 - 1) // BM2
        coff2 = tl.cumsum(chunks2, 0) - chunks2
        total2 = tl.sum(chunks2, 0)
        w2 = tl.arange(0, W_MAX2)
        sel2 = (coff2[None, :] <= w2[:, None]) & ((coff2 + chunks2)[None, :] > w2[:, None])
        owner2 = tl.sum(tl.where(sel2, e_ar[None, :], 0), 1)
        c_first2 = tl.sum(tl.where(sel2, coff2[None, :], 0), 1)
        e_start2 = tl.sum(tl.where(sel2, starts[None, :], 0), 1)
        e_cnt2 = tl.sum(tl.where(sel2, counts[None, :], 0), 1)
        q0 = e_start2 + (w2 - c_first2) * BM2
        q1 = tl.minimum(q0 + BM2, e_start2 + e_cnt2)
        wm2 = w2 < total2
        tl.store(work2_ptr + w2 * 3 + 0, owner2, mask=wm2)
        tl.store(work2_ptr + w2 * 3 + 1, q0, mask=wm2)
        tl.store(work2_ptr + w2 * 3 + 2, q1, mask=wm2)
        tl.store(nwork2_ptr, total2)


_DECODE_ASM = gl.constexpr("""
{
.reg .b32 t, u;
and.b32 t, $4, 0x81C081C0;
mul.rn.bf16x2 $0, t, $6;
shf.l.wrap.b32 t, $4, $4, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $1, t, $6;
shf.l.wrap.b32 t, $4, $4, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $2, t, $6;
shf.l.wrap.b32 t, $4, $4, 6;
shf.l.wrap.b32 u, $4, $4, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $3, t, $6;
}
""")
_DECODE_ASM16 = gl.constexpr("""
{
.reg .b32 t, u;
and.b32 t, $8, 0x81C081C0;
mul.rn.bf16x2 $0, t, $24;
and.b32 t, $10, 0x81C081C0;
mul.rn.bf16x2 $1, t, $25;
shf.l.wrap.b32 t, $8, $8, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $2, t, $24;
shf.l.wrap.b32 t, $10, $10, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $3, t, $25;
shf.l.wrap.b32 t, $8, $8, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $4, t, $24;
shf.l.wrap.b32 t, $10, $10, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $5, t, $25;
shf.l.wrap.b32 t, $8, $8, 6;
shf.l.wrap.b32 u, $8, $8, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $6, t, $24;
shf.l.wrap.b32 t, $10, $10, 6;
shf.l.wrap.b32 u, $10, $10, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $7, t, $25;
}
""")


# Layouts (CTA of 4 warps, BN = 64 weight rows, BM = 16 token rows, BK = 256).
MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA_L.value, k_width=2))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MMA_L.value, k_width=2))
# packed words [BN rows, 4 (q), 8 (j)]: j in registers (contiguous 32 bytes), q in lanes
L3 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 8]))
# Decode input/output [BN rows, 4 (q), 8 (word), 2 (b0), 2 (b2), 2 (b3)]: the 16 elements of one asm call are, in
# register order, (b0, row + 8, b2, b3): the MMA A quads of two k16 steps for rows r and r + 8.
LW16 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]],
    lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
    warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 8, 2, 2, 2]))
# The same decode for one K chunk of 64: words [BN, 4 (q), 2 (word)], so a chunk's decoded A operand is 16
# registers per thread instead of 64.
LW16C = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]],
    lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
    warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 2, 2, 2, 2]))
# block-scale bytes [BN, 4 (q), 4 (g)]
LS = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
# activations [BM token rows, 4 (q), 64 (8j + i)]
LX = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 64]))
# The same activation tile viewed as [64 rows (4n + q), 64]: per-warp read layout from shared memory ...
LX2 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 64]))
LX32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0], [16, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 64]))
LX2_32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 64]))
LX8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 64]))
LX2_8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 64]))
# ... cooperative global load layout (each thread 16 contiguous bytes) ...
LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
# ... and the shared buffer, 16-byte chunks XOR-swizzled by row so a quarter warp's rows hit distinct banks.
X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))
# Shared-B mode: each K block of activations is stored once as [BM, BK] in the MMA's logical K order, 128-byte
# swizzled. On sm_90 (H200) the GEMMs read it with wgmma, the decoded weights being wgmma's register A operand (the
# k_width-2 operand layout below is the same register fragment); elsewhere mma_v2 reads it.
XB_SMEM = gl.constexpr(gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2))
MMA3_8 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16]))
MMA3_16 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 16, 16]))
A3_8 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA3_8.value, k_width=2))
A3_16 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA3_16.value, k_width=2))


@gluon.jit
def _idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2),
            gl.expand_dims(gl.expand_dims(c, 0), 1))


@gluon.jit
def _x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)


@gluon.jit
def _decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, L3: gl.constexpr, A_L: gl.constexpr):
    """Packed words [BN, 4, 8] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 8])
    w16 = gl.convert_layout(_x8(w3), LW16)
    s16 = gl.convert_layout(_x8(s8), LW16)
    t = gl.inline_asm_elementwise(_DECODE_ASM16, "=r,=r,=r,=r,=r,=r,=r,=r," + ",".join(["r"] * 24), [w16, s16],
                                  dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, A_L, assert_trivial=True)


@gluon.jit
def _wsplit4(w3, BN: gl.constexpr):
    """Packed words [BN, 4, 8] (word w = 4 c1 + 2 c0 + wl) -> the four K chunks c = 2 c1 + c0, each [BN, 4, 2]."""
    lo, hi = gl.split(gl.permute(gl.reshape(w3, [BN, 4, 2, 2, 2]), (0, 1, 4, 2, 3)))
    c0, c2 = gl.split(lo)
    c1, c3 = gl.split(hi)
    return c0, c1, c2, c3


@gluon.jit
def _ssplit4(sb, BN: gl.constexpr):
    """Scale bytes [BN, 4, 4] (group g = chunk c = 2 c1 + c0) -> four [BN, 4, 1]."""
    lo, hi = gl.split(gl.reshape(sb, [BN, 4, 2, 2]))
    c0, c2 = gl.split(lo)
    c1, c3 = gl.split(hi)
    return gl.expand_dims(c0, 2), gl.expand_dims(c1, 2), gl.expand_dims(c2, 2), gl.expand_dims(c3, 2)


@gluon.jit
def _mma_chunk(wc, sc, xb_s, C: gl.constexpr, acc, BN: gl.constexpr, a_l: gl.constexpr, HOP: gl.constexpr):
    """Decode K chunk C (64 logical K) and multiply it with the matching 64 columns of the shared activation tile."""
    s = (sc.to(gl.float8e4nv, bitcast=True).to(gl.float32) * SCALE_BIAS).to(gl.bfloat16)
    s2 = gl.reshape(gl.join(s, s), [BN, 4, 2])
    w16 = gl.convert_layout(_x8(wc), LW16C)
    s16 = gl.convert_layout(_x8(s2), LW16C)
    t = gl.inline_asm_elementwise(_DECODE_ASM16, "=r,=r,=r,=r,=r,=r,=r,=r," + ",".join(["r"] * 24), [w16, s16],
                                  dtype=gl.bfloat16, is_pure=True, pack=16)
    a = gl.convert_layout(gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, 64]), a_l)
    bv = xb_s.slice(64 * C, 64, dim=1).permute((1, 0))
    if HOP:
        acc = warpgroup_mma(a, bv, acc)
    else:
        acc = mma_v2(a, bv.load(B_L), acc)
    return acc


@gluon.jit
def _xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, LXG))
    cc = gl.arange(0, 64, layout=gl.SliceLayout(0, LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + (rr % 4) * 64, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return off, msk


@gluon.jit
def _x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(LX2_8), [BM, 4, 64])
    elif BM == 32:
        xv = gl.reshape(x_s.load(LX2_32), [BM, 4, 64])
    else:
        xv = gl.reshape(x_s.load(LX2), [BM, 4, 64])
    return _x_tile(xv, BM, BK, B_L)


@gluon.jit
def _xb_store(x_ptr, off, msk, k0, xb_s, BM: gl.constexpr, BK: gl.constexpr):
    """Cooperative load of one K block of activations, stored to xb_s as [n, k' = 32 w + 16 b3 + 8 b2 + 2 q + b0]."""
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)  # [4 BM, 64]: row 4 n + q, column 8 w + 4 b3 + 2 b2 + b0
    xl = gl.reshape(gl.permute(gl.reshape(xg, [BM, 4, 8, 2, 2, 2]), (0, 2, 3, 4, 1, 5)), [BM, BK])
    gl.barrier()
    xb_s.store(xl)
    fence_async_shared()
    gl.barrier()


@gluon.jit
def _x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, B_L: gl.constexpr):
    # [n, q, 64 physical = 8 w + 4 b3 + 2 b2 + b0] -> [k' = 32 w + 16 b3 + 8 b2 + 2 q + b0, n]
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 8, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, B_L, assert_trivial=BM != 32)


@gluon.jit
def _w13_silu(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, IL: gl.constexpr = False,
              SPLIT: gl.constexpr = 1):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    KS: gl.constexpr = K // SPLIT  # with SPLIT > 1 (M = 1 only) program (., ., k) covers K part k and stores
    pid_k = gl.program_id(2) if SPLIT > 1 else 0  # fp32 gate/up partial sums; _d1_reduce adds them
    kb = pid_k * KS
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if pid_n + pid_k == 0:  # this slot's local expert id, or -1, for the post-expert norm
            gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
        BLKW: gl.constexpr = BN * 36 if IL else BN * 32  # IL: codes then scales per 9216 B block
        BLKB: gl.constexpr = BN * 144 if IL else BN * 16
        S0: gl.constexpr = BN * 128 if IL else 0
        w_off = (e * (2 * I // BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (2 * I // BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
            wg_n = gl.load(wp + w_off + (kb // BK) * BLKW)
            sg_n = gl.load(sf_ptr + s_off + (kb // BK) * BLKB)
            wu_n = gl.load(wp + w_off + (I // BN) * KB * BLKW + (kb // BK) * BLKW)
            su_n = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB + (kb // BK) * BLKB)
        for k0 in range(kb, kb + KS, BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + BK
                wm = (rw >= 0) & (kn < kb + KS)
                sm = (rs >= 0) & (kn < kb + KS)
                wg_n = gl.load(wp + w_off + (kn // BK) * BLKW, mask=wm, other=0)
                sg_n = gl.load(sf_ptr + s_off + (kn // BK) * BLKB, mask=sm)
                wu_n = gl.load(wp + w_off + (I // BN) * KB * BLKW + (kn // BK) * BLKW, mask=wm, other=0)
                su_n = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB + (kn // BK) * BLKB, mask=sm)
            else:
                wg = gl.load(wp + w_off + (k0 // BK) * BLKW)
                sg = gl.load(sf_ptr + s_off + (k0 // BK) * BLKB)
                wu = gl.load(wp + w_off + (I // BN) * KB * BLKW + (k0 // BK) * BLKW)
                su = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB + (k0 // BK) * BLKB)
            if SMB:
                _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    g0, g1, g2, g3 = _wsplit4(wg, BN)
                    h0, h1, h2, h3 = _ssplit4(sg, BN)
                    acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                    u0, u1, u2, u3 = _wsplit4(wu, BN)
                    v0, v1, v2, v3 = _ssplit4(su, BN)
                    acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
                elif HOP:
                    acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(B_L)
                    acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                    acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
            else:
                if XS:
                    xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        omask = gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1)
        if SPLIT > 1:  # fp32 partials [SPLIT, TOP_K, 2 I], gate columns then up columns
            pptr = out_ptr + pid_k * (8 * 2 * I) + gl.expand_dims(pair_o, 0) * (2 * I) + gl.expand_dims(n0 + n_o, 1)
            gl.store(pptr, acc_g, mask=omask)
            gl.store(pptr + I, acc_u, mask=omask)
        else:
            g = gl.load(gs_ptr + e) * ACC_BIAS
            hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
            hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
            inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
            optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, inter.to(gl.bfloat16), mask=omask)


@gluon.jit
def _gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))


@gluon.jit
def _w2(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
        K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
        SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, GU: gl.constexpr = False,
        IL: gl.constexpr = False):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if GU:
            if pid_n == 0:  # this slot's local expert id, or -1, for the post-expert norm
                gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights
        BLKW: gl.constexpr = BN * 36 if IL else BN * 32
        BLKB: gl.constexpr = BN * 144 if IL else BN * 16
        S0: gl.constexpr = BN * 128 if IL else 0
        w_off = (e * (N // BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (N // BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8 if GU else pair) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, GU, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:
            wq_n = gl.load(wp + w_off)
            sq_n = gl.load(sf_ptr + s_off)
        for k0 in range(0, K, BK):
            if PF:
                wq = wq_n
                sq = sq_n
                kn = k0 + BK
                wq_n = gl.load(wp + w_off + (kn // BK) * BLKW, mask=(rw >= 0) & (kn < K), other=0)
                sq_n = gl.load(sf_ptr + s_off + (kn // BK) * BLKB, mask=(rs >= 0) & (kn < K))
            else:
                wq = gl.load(wp + w_off + (k0 // BK) * BLKW)
                sq = gl.load(sf_ptr + s_off + (k0 // BK) * BLKB)
            if SMB:
                _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    q0, q1, q2, q3 = _wsplit4(wq, BN)
                    p0, p1, p2, p3 = _ssplit4(sq, BN)
                    acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
                elif HOP:
                    acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
                else:
                    acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
            else:
                if XS:
                    xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
        y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        if GU:  # rows 16 s + i are gate column 8 s + i of this tile's 32, rows 16 s + 8 + i its up rows
            rows2 = gl.expand_dims(n_o, 1) + gl.expand_dims(m_o * 0, 0)
            pair2 = gl.expand_dims(pair_o, 0) + gl.expand_dims(n_o * 0, 1)
            msk2 = gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1)
            hg, hu = _gu_split(y, BN, BM)
            rg, _ru = _gu_split(rows2, BN, BM)
            pg, _pu = _gu_split(pair2, BN, BM)
            mg, _mu = _gu_split(msk2.to(gl.int32), BN, BM)
            hg = hg.to(gl.bfloat16).to(gl.float32)
            hu = hu.to(gl.bfloat16).to(gl.float32)
            v = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
            col = pid_n * (BN // 2) + (rg // 16) * 8 + rg % 8
            gl.store(out_ptr + pg * (N // 2) + col, v.to(gl.bfloat16), mask=mg != 0)
        else:
            optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))
    if DIRECT and not GU:
        if r1 == r0:  # non-local slot: zero output row, which the post-expert norm maps to zero
            zl: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
            zo = gl.arange(0, BN, layout=zl)
            gl.store(out_ptr + pid_w * N + pid_n * BN + zo, gl.zeros([BN], gl.bfloat16, layout=zl))


# ---- split-K W2 at decode 1 ----
# _w2_ks splits decode-1 W2 over K into KS CTAs writing fp32 partials; _wsum_ks is SGLang's post-expert norm weighted
# sum (same math, order and rsqrt) that first adds the KS partials and rounds once to BF16.
@gluon.jit
def _w2_ks(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
           K: gl.constexpr, N: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, KS: gl.constexpr,
           PF: gl.constexpr, NLOC: gl.constexpr = 32):
    BN: gl.constexpr = 64
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    pid_k = gl.program_id(2)
    e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
    active = (e >= 0) & (e < NLOC)
    e = e * active.to(gl.int32)
    r0 = pid_w
    r1 = pid_w + active.to(gl.int32)
    if active:
        KB: gl.constexpr = K // BK
        KSB: gl.constexpr = KB // KS  # K blocks per split
        kb0 = pid_k * KSB
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        w_off = (e * (N // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (N // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, False, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=MMA_L)
        if PF:
            wq_n = gl.load(wp + w_off + kb0 * (BN * 32))
            sq_n = gl.load(sf_ptr + s_off + kb0 * (BN * 16))
        for j in range(0, KSB):
            kb = kb0 + j
            if PF:
                wq = wq_n
                sq = sq_n
                more = j + 1 < KSB
                wq_n = gl.load(wp + w_off + (kb + 1) * (BN * 32), mask=(rw >= 0) & more, other=0)
                sq_n = gl.load(sf_ptr + s_off + (kb + 1) * (BN * 16), mask=(rs >= 0) & more)
            else:
                wq = gl.load(wp + w_off + kb * (BN * 32))
                sq = gl.load(sf_ptr + s_off + kb * (BN * 16))
            xb = _x_shared(a_ptr, xoff, xmsk, kb * BK, x_s, BM, BK)
            acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
        y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, MMA_L))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, MMA_L))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + pid_k * (8 * N) + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(pid_n * BN + n_o, 1)
        gl.store(optr, y, mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@triton.jit
def _wsum_ks(Y, NW, TW, OUT, IDS, eps, H: tl.constexpr, KS: tl.constexpr, TOPK: tl.constexpr, FP32: tl.constexpr):
    """SGLang's _post_expert_norm_weighted_sum_kernel (APPLY_NORM, HAS_WEIGHT, WEIGHT_FIRST, HAS_IDS; same math,
    order and rsqrt), with one change: FP32 means Y holds KS fp32 partial rows per slot, summed and rounded to BF16
    (as W2's BF16 output is) before the unchanged norm."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(NW + offs).to(tl.float32)
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        live = tl.load(IDS + t * TOPK + k) >= 0
        if live:
            if FP32:
                x = tl.zeros([H], tl.float32)
                for s in tl.static_range(KS):
                    x += tl.load(Y + s * (TOPK * H) + (t * TOPK + k) * H + offs)
                x = x.to(tl.bfloat16).to(tl.float32)
            else:
                x = tl.load(Y + (t * TOPK + k) * H + offs).to(tl.float32)
            y = x * w
            var = tl.sum(y * y, axis=0) / H
            y = y * tl.rsqrt(var + eps)
            acc += y * tl.load(TW + t * TOPK + k).to(tl.float32)
    tl.store(OUT + t * H + offs, acc.to(OUT.dtype.element_ty))


@triton.jit
def _d1_reduce(P, GS, IDS, INTER, I: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, NLOC: tl.constexpr):
    """M = 1: inter[slot] = SiLU(gate) * up from W13's S fp32 partials [S, 8, 2I] of each local slot, with
    W13's own scale and roundings. Program (slot, column block); non-local slots return."""
    slot = tl.program_id(0)
    lid = tl.load(IDS + slot)
    if (lid >= 0) & (lid < NLOC):
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ga = tl.zeros([BLOCK], tl.float32)
        ua = tl.zeros([BLOCK], tl.float32)
        for s in tl.static_range(S):
            ga += tl.load(P + s * (8 * 2 * I) + slot * (2 * I) + c)
            ua += tl.load(P + s * (8 * 2 * I) + slot * (2 * I) + I + c)
        g = tl.load(GS + lid) * 128.0  # ACC_BIAS
        hg = (ga * g).to(tl.bfloat16).to(tl.float32)
        hu = (ua * g).to(tl.bfloat16).to(tl.float32)
        v = (hg / (1.0 + tl.exp(-hg))).to(tl.bfloat16).to(tl.float32) * hu
        tl.store(INTER + slot * I + c, v.to(tl.bfloat16))


# Verify-path kernels whose CTAs pick their token tile per work item. An item of at most 8 rows
# (most experts' last item at verify) runs the 8-row tile: half the MMAs, X staging and epilogue of a 16-row tile,
# with the same weight stream and decode. The bodies are _w13_silu's and _w2's non-DIRECT bodies.
@gluon.jit
def _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr, HOP: gl.constexpr, CHK: gl.constexpr, PF: gl.constexpr):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    n0 = pid_n * BN
    rw, qw, jw = _idx3(BN, 4, 8, L3)
    KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
    w_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _idx3(BN, 4, 4, LS)
    s_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
    rv = r0 + mx
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = (pair // 8) * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
    xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
    wp = w_ptr.to(gl.pointer_type(gl.int32))
    acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
    acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
    if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
        wg_n = gl.load(wp + w_off)
        sg_n = gl.load(sf_ptr + s_off)
        wu_n = gl.load(wp + w_off + I * (K // 8))
        su_n = gl.load(sf_ptr + s_off + I * (K // 16))
    for k0 in range(0, K, BK):
        if PF:
            wg = wg_n
            sg = sg_n
            wu = wu_n
            su = su_n
            kn = k0 + BK
            wm = (rw >= 0) & (kn < K)
            sm = (rs >= 0) & (kn < K)
            wg_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=wm, other=0)
            sg_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=sm)
            wu_n = gl.load(wp + w_off + I * (K // 8) + (kn // BK) * (BN * 32), mask=wm, other=0)
            su_n = gl.load(sf_ptr + s_off + I * (K // 16) + (kn // BK) * (BN * 16), mask=sm)
        else:
            wg = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
            sg = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
            wu = gl.load(wp + w_off + I * (K // 8) + (k0 // BK) * (BN * 32))
            su = gl.load(sf_ptr + s_off + I * (K // 16) + (k0 // BK) * (BN * 16))
        if SMB:
            _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
            if CHK:
                g0, g1, g2, g3 = _wsplit4(wg, BN)
                h0, h1, h2, h3 = _ssplit4(sg, BN)
                acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                u0, u1, u2, u3 = _wsplit4(wu, BN)
                v0, v1, v2, v3 = _ssplit4(su, BN)
                acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
            elif HOP:
                acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
            else:
                xb = xb_s.permute((1, 0)).load(B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        else:
            if XS:
                xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
            else:
                xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
            acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
            acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
    g = gl.load(gs_ptr + e) * ACC_BIAS
    hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
    hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
    inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
    n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
    gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@gluon.jit
def _w13_mix(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
             K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
             XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
             CHK: gl.constexpr = False, PF: gl.constexpr = False):
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    active = pid_w < gl.load(nwork_ptr)
    e = gl.load(work_ptr + pid_w * 3 + 0)
    r0 = gl.load(work_ptr + pid_w * 3 + 1)
    r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                      K=K, I=I, BN=BN, BK=BK, BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        elif BM == 32:
            if r1 - r0 <= 16:
                _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                          K=K, I=I, BN=BN, BK=BK, BM=16, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
            else:
                _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                          K=K, I=I, BN=BN, BK=BK, BM=32, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                      K=K, I=I, BN=BN, BK=BK, BM=BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)


@gluon.jit
def _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
             K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
             XS: gl.constexpr, SMB: gl.constexpr, HOP: gl.constexpr, CHK: gl.constexpr, PF: gl.constexpr):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    n0 = pid_n * BN
    rw, qw, jw = _idx3(BN, 4, 8, L3)
    KB: gl.constexpr = K // BK  # tile-major weights
    w_off = (e * (N // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _idx3(BN, 4, 4, LS)
    s_off = (e * (N // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
    rv = r0 + mx
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = pair * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, False, BM)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
    xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
    wp = w_ptr.to(gl.pointer_type(gl.int32))
    acc = gl.zeros([BN, BM], gl.float32, layout=mma)
    if PF:
        wq_n = gl.load(wp + w_off)
        sq_n = gl.load(sf_ptr + s_off)
    for k0 in range(0, K, BK):
        if PF:
            wq = wq_n
            sq = sq_n
            kn = k0 + BK
            wq_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=(rw >= 0) & (kn < K), other=0)
            sq_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=(rs >= 0) & (kn < K))
        else:
            wq = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
            sq = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
        if SMB:
            _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
            if CHK:
                q0, q1, q2, q3 = _wsplit4(wq, BN)
                p0, p1, p2, p3 = _ssplit4(sq, BN)
                acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
            elif HOP:
                acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
            else:
                acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
        else:
            if XS:
                xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
            else:
                xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
            acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
    y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
    n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
    gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@gluon.jit
def _w2_mix(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
            K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
            SMB: gl.constexpr = False, HOP: gl.constexpr = False,
            CHK: gl.constexpr = False, PF: gl.constexpr = False):
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    active = pid_w < gl.load(nwork_ptr)
    e = gl.load(work_ptr + pid_w * 3 + 0)
    r0 = gl.load(work_ptr + pid_w * 3 + 1)
    r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                     K=K, N=N, BN=BN, BK=BK, BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                     K=K, N=N, BN=BN, BK=BK, BM=BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)



def _src_bit(j, t):
    """Source bit in the stored word of target bit t of decoded output j (see _DECODE_ASM)."""
    if j == 0:
        return t
    if j == 1:
        return (t - 4) % 32
    if j == 2:
        return (t - 10) % 32
    return (t - 6) % 32 if (0x80C080C0 >> t) & 1 else (t - 14) % 32


def _permute_bits(packed):
    """uint8 [..., K/2] (two E2M1 codes per byte, low nibble first) -> the same codes with each 32-bit word's bits
    permuted (lossless): natural code c of the word goes to decoded output c // 2, half c % 2."""
    words = packed.contiguous().view(torch.int32)
    out = torch.zeros_like(words)
    for c in range(8):
        j, h = c // 2, c % 2
        for bit, tgt in enumerate((6, 7, 8, 15)):
            out |= ((words >> (4 * c + bit)) & 1) << _src_bit(j, tgt + 16 * h)
    return out.view(torch.uint8).view(packed.shape)


def _tile_major(codes, sf):
    """Codes uint8 [L, rows, K/2] and fp8 scales [L, rows, K/16] -> the same bytes ordered tile-major, so that
    each (expert, BN-row tile, BK block) is one contiguous BN x 128 B run of codes and one BN x 16 B run of scales
    (a lossless permutation of bytes; contiguous blocks stream from DRAM better than BN rows 2 KB apart)."""
    n_l, rows, half_k = codes.shape
    kb = 2 * half_k // BK
    c = codes.view(n_l, rows // BN, BN, kb, BK // 2).permute(0, 1, 3, 2, 4).contiguous()
    s = sf.contiguous().view(n_l, rows // BN, BN, kb, BK // 16).permute(0, 1, 3, 2, 4).contiguous()
    return c, s


def _gu_rows(I):
    """Row order of the gate/up-paired W13 copy. New row 64 t + 16 s + 8 g + i holds gate (g = 0) or up (g = 1)
    row of output column 32 t + 8 s + i (a lossless row permutation)."""
    r = torch.arange(2 * I)
    t, s, g, i = r // 64, (r % 64) // 16, (r % 16) // 8, r % 8
    return g * I + 32 * t + 8 * s + i


def _interleaved(c, s):
    """Tile-major codes and scales merged per block (codes then scales, 9216 B), a second lossless copy
    used from IL_MIN_M tokens and at IL_M, where it streams better."""
    n_l, nt, kb = c.shape[:3]
    blk = torch.cat([c.reshape(n_l, nt, kb, -1), s.view(torch.uint8).reshape(n_l, nt, kb, -1)], dim=3).contiguous()
    return blk, blk.view(torch.float8_e4m3fn)


def prepare(weights):
    # This GPU's experts [0, 32): codes bit-permuted within 32-bit words (lossless); scales unchanged.
    L = 32
    w13, w13_sf = _tile_major(torch.stack([_permute_bits(weights["w13"][e]) for e in range(L)]), weights["w13_sf"][:L])
    gu = _gu_rows(weights["w13"].shape[1] // 2).to(weights["w13"].device)
    w13g, w13g_sf = _tile_major(torch.stack([_permute_bits(weights["w13"][e][gu]) for e in range(L)]),
                                weights["w13_sf"][:L][:, gu])
    w2, w2_sf = _tile_major(torch.stack([_permute_bits(weights["w2"][e]) for e in range(L)]), weights["w2_sf"][:L])
    state = {
        "w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"][:L].float().contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"][:L].float().contiguous(),
        "w13g": w13g, "w13g_sf": w13g_sf,
        "il13": _interleaved(w13, w13_sf), "il2": _interleaved(w2, w2_sf),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "local": (0, L),
        "hopper": torch.cuda.get_device_capability()[0] == 9,
        "local_start_t": torch.full((1 + TOP_K,), -1, dtype=torch.int32, device=weights["w13"].device),
        "iota8": torch.arange(TOP_K, dtype=torch.int32, device=weights["w13"].device),
    }
    state["local_start_t"][0] = 0  # entry 0: global id of local expert 0; entries 1..8: per-slot local ids (written by W13)
    torch.cuda.synchronize()
    return state


def _decode1_direct(x, ids, tw, state, H, I, L):
    """M = 1: three launches (W13 + SiLU, W2, post-expert norm); each top-k slot is one item, no routing pass."""
    dev = x.device
    inter = torch.empty((TOP_K, I), dtype=torch.bfloat16, device=dev)
    if 1 in GU_M:  # gate/up-paired W13, one weight stream per CTA
        _w2[(2 * I // BN, TOP_K)](x, state["w13g"], state["w13g_sf"], state["w13_gs"], state["iota8"], ids,
                                  state["local_start_t"], inter, K=H, N=2 * I, BN=BN, BK=BK, BM=8, XS=True, SMB=False,
                                  HOP=False, CHK=False, PF=True, DIRECT=True, NLOC=L, GU=True, num_warps=NUM_WARPS)
    else:
        S = D1_W13_SPLIT  # W13 split over K, partials reduced by a small kernel
        part = torch.empty((S, TOP_K, 2 * I), dtype=torch.float32, device=dev)
        _w13_silu[(I // BN, TOP_K, S)](x, state["w13"], state["w13_sf"], state["w13_gs"], state["iota8"], ids,
                                       state["local_start_t"], part, K=H, I=I, BN=BN, BK=BK, BM=8, XS=True, SMB=False,
                                       HOP=False, CHK=False, PF=True, DIRECT=True, NLOC=L, SPLIT=S,
                                       num_warps=NUM_WARPS)
        _d1_reduce[(TOP_K, I // 1024)](part, state["w13_gs"], state["local_start_t"][1:], inter, I=I, S=S,
                                       BLOCK=1024, NLOC=L, num_warps=4)
    ks = D1_W2_KS  # W2 split over K, fp32 partials summed inside the weighted sum
    yp = torch.empty((ks, TOP_K, H), dtype=torch.float32, device=dev)
    _w2_ks[(H // BN, TOP_K, ks)](inter, state["w2"], state["w2_sf"], state["w2_gs"], state["iota8"], ids,
                                 state["local_start_t"], yp, K=I, N=H, BK=BK, BM=8, KS=ks, PF=True, NLOC=L,
                                 num_warps=NUM_WARPS)
    out = torch.empty((1, H), dtype=torch.bfloat16, device=dev)
    _wsum_ks[(1,)](yp, state["norm_w"], tw, out, state["local_start_t"][1:], POST_NORM_EPS, H=H, KS=ks, TOPK=TOP_K,
                   FP32=True, num_warps=8)
    return out


def _w13_regcap(M):
    n = (W13_MAXNREG_MIX if M >= MIX_MIN_M else W13_MAXNREG_BIG) if M >= SMB_MIN_M else W13_MAXNREG.get(M)
    return {"maxnreg": n} if n else {}


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M, H = x.shape
    I = cfg["d_expert"]
    L = cfg["local_count"]
    ls = cfg["local_start"]
    assert (ls, L) == state["local"], "prepared for experts [0, 32) (ep8, this GPU)"
    dev = x.device
    P = M * TOP_K
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].contiguous()
    if M == 1:
        return _decode1_direct(x, ids, tw, state, H, I, L)
    p_block = triton.next_power_of_2(P)
    bm = 8 if M <= 32 else BM
    w_max = triton.next_power_of_2(triton.cdiv(P, bm) + L)
    local = torch.empty((M, TOP_K), dtype=torch.int32, device=dev)
    sorted_pairs = torch.empty((P,), dtype=torch.int32, device=dev)
    work = torch.empty((w_max, 3), dtype=torch.int32, device=dev)
    nwork = torch.empty((1,), dtype=torch.int32, device=dev)
    grid_w = triton.cdiv(P, bm) + L
    if M >= MIX_MIN_M:  # one routing pass, two work lists (W2's 16-row items, W13's BM_W13-row items)
        bm13 = BM_W13
        grid_w13 = triton.cdiv(P, bm13) + L
        w_max13 = triton.next_power_of_2(grid_w13)
        work13 = torch.empty((w_max13, 3), dtype=torch.int32, device=dev)
        nwork13 = torch.empty((1,), dtype=torch.int32, device=dev)
        _route2_mc[(L + 3,)](ids, local, sorted_pairs, work, nwork, work13, nwork13, P, ls, LOCAL=L, BM=bm, BM2=bm13,
                             P_BLOCK=p_block, W_MAX=w_max, W_MAX2=w_max13, num_warps=16 if p_block >= 1024 else 4)
    else:
        _route[(1,)](ids, local, sorted_pairs, work, nwork, P, ls, LOCAL=L, BM=bm, P_BLOCK=p_block, W_MAX=w_max,
                     num_warps=16 if p_block >= 1024 else 4)
    inter = torch.empty((P, I), dtype=torch.bfloat16, device=dev)
    il = M in IL_M  # interleaved blocks at M = 8 only (they do not combine with the per-item verify kernels at M >= 128)
    w13c, w13s = state["il13"] if il else (state["w13"], state["w13_sf"])
    w2c, w2s = state["il2"] if il else (state["w2"], state["w2_sf"])
    if M in GU_M:  # gate/up-paired W13 at these token counts
        _w2[(2 * I // BN, grid_w)](x, state["w13g"], state["w13g_sf"], state["w13_gs"], sorted_pairs, work, nwork,
                                   inter, K=H, N=2 * I, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=False, HOP=False,
                                   CHK=True, PF=M <= PF_MAX_M, GU=True, num_warps=NUM_WARPS)
    elif M >= MIX_MIN_M:  # W13 on 32-row items, per-item tile, register path
        smb = VERIFY_SMB and M >= SMB_MIN_M
        _w13_mix[(I // BN, grid_w13)](x, state["w13"], state["w13_sf"], state["w13_gs"], sorted_pairs, work13, nwork13,
                                      inter, K=H, I=I, BN=BN, BK=BK, BM=bm13, XS=M >= XS_MIN_M, SMB=smb,
                                      HOP=state["hopper"] and smb, CHK=True, PF=M <= PF_MAX_M,
                                      num_warps=NUM_WARPS, **_w13_regcap(M))
    else:
        _w13_silu[(I // BN, grid_w)](x, w13c, w13s, state["w13_gs"], sorted_pairs, work, nwork, inter,
                                     K=H, I=I, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=M >= SMB_MIN_M,
                                     HOP=state["hopper"] and M >= SMB_MIN_M, CHK=True, PF=M <= PF_MAX_M,
                                     IL=il, num_warps=NUM_WARPS, **_w13_regcap(M))
    y = torch.empty((P, H), dtype=torch.bfloat16, device=dev)
    if M >= MIX_MIN_M and not W2_VERIFY_C175:  # W2 on the register path with the per-item tile
        smb2 = VERIFY_SMB and M >= SMB_MIN_M
        _w2_mix[(H // BN, grid_w)](inter, state["w2"], state["w2_sf"], state["w2_gs"], sorted_pairs, work, nwork, y,
                                   K=I, N=H, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=smb2,
                                   HOP=state["hopper"] and smb2, CHK=True, PF=M <= PF_MAX_M, num_warps=NUM_WARPS)
    else:
        _w2[(H // BN, grid_w)](inter, w2c, w2s, state["w2_gs"], sorted_pairs, work, nwork, y,
                               K=I, N=H, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=M >= SMB_MIN_M,
                               HOP=state["hopper"] and M >= SMB_MIN_M, CHK=True, PF=M <= PF_MAX_M,
                               IL=il, num_warps=NUM_WARPS)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    block_h = triton.next_power_of_2(H)
    _post_expert_norm_weighted_sum_kernel[(M,)](
        y, state["norm_w"], tw, out, local, TOPK=TOP_K, H=H, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4, num_stages=2)
    return out


