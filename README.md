# Open Kernel Challenge Solutions (NVIDIA H200 / SM90a)

High-performance GPU Megakernels written for the **Open Kernel Challenge** ([openkernelchallenge.dev](https://openkernelchallenge.dev/)), accelerating Meta's **Muse Spark Open (816B MoE NVFP4)** foundation model on NVIDIA H200 SXM GPUs (`sm_90a`, 4.8 TB/s HBM3e, 141 GB VRAM, 132 SMs).

Evaluated under strict CUDA Graphs with cold L2 cache enforcement against SGLang's baseline kernels (`lmsysorg/sglang:dev-muse-spark-open`).

---

## Leaderboard & Official Results (Team `ruwwww` / Committer `kobeniw`)

**Overall Global Standing:** **Rank #167–#171 Worldwide** (out of 1,150+ teams).

| Board / Problem | SGLang Baseline | Optimized Latency | Official Speedup | World Rank | Official Job ID | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`muse_attention_decode`** | 3,766.3 µs | **1,446.8 µs** | **`2.6032x`** | **Rank #69** 🏆 | `2827c6e61a784868` | **PASSED** (16/16) |
| **`muse_moe_w4a16`** | 13,448.5 µs | **5,754.6 µs** | **`2.3370x`** | **Rank #85** 🏆 | `3ed21907cbff49be` | **PASSED** (5/5) |
| **`muse_prefill_layer`** | 387,403.9 µs | **217,625.5 µs** | **`1.7801x`** | **Rank #91** 🏆 | `466e2943491845bf` | **PASSED** (5/5) |
| **`muse_decode_layer`** | 22,620.3 µs | **11,298.5 µs** | **`2.0021x`** | **Rank #167** 🏆 | `5e7cbd571cc54e31` | **PASSED** (13/13) |
| **`muse_bf16_proj`** | 4,830.8 µs | **3,208.6 µs** | **`1.5056x`** | **Rank #206** 🏆 | `c512f73af0454fa2` | **PASSED** (9/9) |

---

## Architectural Breakthroughs Across All 5 Problems

### 1. `muse_attention_decode` (2.6032x Speedup — Rank #69)
- **Single-Kernel End-to-End Fusion:** Replaces 4 disjoint passes (Triton QKNorm/RoPE $\to$ KV store $\to$ FlashAttention-3 $\to$ RMSNorm/Gate) with **one single CUDA Megakernel per call**.
- **WGMMA FP8 Tensor Cores:** $QK^T$ evaluates on FP8 Tensor Core MMAs (queries quantized to E4M3 online) and $PV$ computes on FP8 WGMMA on SM90a.
- **Asynchronous Paged Cache Streaming:** Warps asynchronously stream paged FP8 key/values via `cp.async` in 64-key tiles with multi-stage software pipelining.
- **Fused Epilogue in Registers:** RMSNorm variance reduction and sigmoid gating occur entirely within registers before writing the final BF16 output, completely eliminating intermediate global memory roundtrips.
- **Key Metric:** Decode 1 latency drops from 20.4 µs to **4.6 µs (4.334x speedup)**.

### 2. `muse_moe_w4a16` (2.3370x Speedup — Rank #85)
- **132 Persistent CTAs (Persistent Grid):** Replaced multi-kernel Triton dispatch with a single native CUDA persistent grid (1 CTA per H200 SM) that stays resident on-chip.
- **Zero-DRAM Intermediate Activations:** Phase A ($W_{13}$ GEMM1 + SwiGLU) and Phase B ($W_2$ GEMM2 + RMSNorm stats) run inside the same persistent CTA loop. SwiGLU results remain in local scratch SMEM/registers and feed directly into $W_2$.
- **CUTLASS SM90 L2 Cache Policies:** Applied `.L2::cache_hint` PTX instructions to prioritize evicting weight lines while retaining active token representations and normalization accumulators in L2.
- **Dynamic Dual Work Queues:** Experts are partitioned into heavy (fat) jobs (streamed along $K$) and skinny jobs (16 rows with split-K warps), dynamically dequeued via atomic counters (`atom.inc`).
- **Key Metric:** Decode 1 latency drops to **24.8 µs (3.102x)**; Verify 32 drops to **220.2 µs (2.845x)**.

### 3. `muse_prefill_layer` (1.7801x Speedup — Rank #91)
- **Dual-Engine Attention Selection:** Dynamically chooses between tuned FA3 paged FP8 attention and a specialized Hopper attention Megakernel (`attn_fwd_kernel`) based on context length and layer type (sliding window vs full attention).
- **Multi-Stream Producer-Consumer Overlap:** Concurrently executes router linear projection and top-k expert selection on a secondary CUDA stream while pooler GEMM and pre-expert normalization run on the main stream, joining via `cudaStreamWaitEvent`.
- **Fused Boundary Kernels (`boundary_kernel`):** Integrates RMSNorm, residual stream mixing (`alpha * stream + beta * rms(branch)`), and MoE work-counter resets in a single contiguous memory pass.

### 4. `muse_decode_layer` (2.0021x Speedup — Rank #167)
- **End-to-End Layer Megakernel Pipeline:** Unifies the full transformer decode step:
  $$\text{Input} \to \text{QKV Proj} \to \text{Gated Attn Megakernel} \to \text{O Proj} \to \text{Boundary 1} \to \text{Pooler/Router} \to \text{MoE Persistent Megakernel} \to \text{Unpooler} \to \text{Boundary 2}$$
- Eliminates CPU launch overhead and device-wide synchronization barriers between attention and MoE blocks under CUDA Graph execution.

### 5. `muse_bf16_proj` (1.5000x Speedup — Rank #216)
- **Memory-Bound Assessment:** In 24 of 30 evaluation cases ($M \le 32$), Operational Intensity is only $1.0\text{ FLOP / Byte} \ll 206\text{ FLOPs / Byte}$ (Ridge Point). The problem is 100% memory-bandwidth bound.
- **Lossless 12-bit Weight Compression (25% Bandwidth Savings):**
  - High-density normal weights are packed into 12 bits (8-bit low byte mantissa + 4-bit high nibble exponent index).
  - Bit-reconstructed on-the-fly in PTX registers with negligible ALU overhead, reducing HBM read traffic by 25%.
  - Out-of-window escapes (<0.2%) are corrected using a sparse FP32 correction table.
- **Skinny Swap-AB Tensor-Core GEMV (`sk2`):** Stages activations $X$ in SMEM as matrix $A$ and streams compressed weights $W$ into registers as matrix $B$ using `mma.m16n8k16`, avoiding cross-CTA atomic reduction.
- **Router Split-K Precision:** Splits $W_{\text{router}}$ into $W_{\text{hi}} + W_{\text{lo}}$ (BF16 each) with FP32 accumulation to satisfy the strict $10^{-4}$ tolerance threshold.

---

## Repository Structure

```
.
├── README.md
├── muse_attention_decode/ # Rank #69 (2.6032x) - Fused CUDA Gated Attention Megakernel
│   ├── submission.py
│   ├── TASK.md
│   └── profiles/          # Remote H200 profile & execution traces
├── muse_moe_w4a16/        # Rank #85 (2.3370x) - Native Persistent MoE Megakernel
│   ├── submission.py
│   ├── TASK.md
│   └── profiles/
├── muse_prefill_layer/    # Rank #91 (1.7801x) - Full Prefill Layer Megakernel
│   ├── submission.py
│   ├── TASK.md
│   └── profiles/
├── muse_decode_layer/     # Rank #167 (2.0021x) - Full Decode Layer Megakernel
│   ├── submission.py
│   ├── TASK.md
│   └── profiles/
├── muse_bf16_proj/        # Rank #216 (1.5000x) - Skinny Swap-AB & 12-bit Weight GEMV
│   ├── submission.py
│   ├── TASK.md
│   └── profiles/
└── serving/
    └── RULES.md           # End-to-end SGLang serving challenge rules
```

---

## Author & Credits

- **Team:** `ruwwww`
- **Committer:** `kobeniw` (`kobeniw@users.noreply.github.com`)
- **Challenge:** [Open Kernel Challenge](https://openkernelchallenge.dev/) (Meta & Andromeda)
