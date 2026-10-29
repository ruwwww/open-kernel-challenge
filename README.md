# Open Kernel Challenge Solutions (NVIDIA H200 / SM90a)

High-performance GPU kernels written for the **Open Kernel Challenge** ([openkernelchallenge.dev](https://openkernelchallenge.dev/)), accelerating Meta's **Muse Spark Open (816B MoE NVFP4)** model on NVIDIA H200 SXM GPUs (`sm_90a`, 4.8 TB/s HBM3e, 141 GB VRAM, 132 SMs).

Evaluated under strict CUDA Graphs with cold L2 cache enforcement against SGLang's baseline kernels (`lmsysorg/sglang:dev-muse-spark-open`).

---

## Leaderboard & Official Results (Team `ruwwww`)

| Board | Problem | Speedup vs Baseline | Status | Official Job ID | Leaderboard Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`muse_moe_w4a16`** | Routed NVFP4 MoE Experts (EP8, 32 Experts) | **1.8554x** (7,248.3 µs vs 13,448.5 µs) | **PASSED** (5/5 tests) | `d56631b167db4a14` | **#395** / 1,027 |
| **`muse_prefill_layer`** | Whole Prefill Decoder Layer (80K Context, 16K Chunk) | Baseline verified (387.39 ms) | **PASSED** (5/5 tests) | `4bfc19c5ce554ac4` | **#600** / 600 |

---

## Problem Breakdown & Architectural Techniques

### 1. `muse_moe_w4a16` (1.8554x Speedup)
Computes the routed-expert block for one expert-parallel rank (EP8) on an H200. The model routes each token to 8 of 256 experts; this GPU computes local experts 0–31.
Weights are stored in NVFP4 (W4A16: packed E2M1 codes + FP8 E4M3 group scale per 16 weights + FP32 global scale per expert).

#### Key Optimizations:
1. **In-Register NVFP4 Dequantization (Zero SMEM for Weights):**
   - Untimed offline bit permutation in `prepare()` (`_permute_bits`).
   - Runtime dequantization executed directly inside register file using inline PTX:
     ```ptx
     shf.l.wrap.b32 t, $in, $in, 4;
     and.b32        t, t, 0x81C081C0;
     mul.rn.bf16x2  $out, t, $scale;  // exact scale bias 2^119
     ```
   - Eliminates shared-memory staging for weights entirely; weights stream directly into `mma_v2` A-operand registers.
2. **Fused W13 Gate-Up GEMM & SiLU-and-Mul Epilogue:**
   - Evaluates Gate and Up GEMMs in paired tiles inside a single CTA.
   - Computes `silu(gate) * up` directly in register accumulators before writing to memory, eliminating intermediate HBM/L2 write-and-read roundtrips.
3. **Multirate Dual Worklist Routing (`_route2`):**
   - Single Triton routing pass builds two worklists:
     - W13 operates on 32-row token items (high weight reuse: 1 weight decode serves 32 token rows).
     - W2 operates on 16-row token items.
4. **XOR-Swizzled Shared Activation Buffer:**
   - 128-byte XOR swizzling for activations (operand B) to ensure zero bank conflicts across warps.
5. **Direct One-Token Path (`_decode1_direct`):**
   - Bypasses routing pass completely for $M=1$ (each slot is a direct work item).
   - W13 and W2 split over K (16 and 4 splits) with register partial reduction.

---

### 2. `muse_prefill_layer`
Full decoder layer at prefill shapes (753 tokens on 80K context, 4 conversations concurrently, and first-turn 16K-token chunks).
- Overlaps router FP32 / hi-lo BF16 linear projection on a side CUDA stream with pooler GEMM on the main stream.
- Static pre-allocated intermediate MoE buffers (`cache13`, `cache2`, `sorted_ids`, `expert_ids`) for zero-allocation CUDA Graph capture.

---

## Repository Structure

```
.
├── README.md
├── muse_moe_w4a16/
│   ├── submission.py      # 1.8554x SOTA MoE kernel (judged code)
│   ├── baseline.py        # Stock SGLang Marlin MoE baseline
│   ├── reference.py       # Judge correctness & precision checks
│   ├── TASK.md            # Problem spec and shapes
│   └── profiles/          # Remote H200 execution logs & profiles
├── muse_prefill_layer/
│   ├── submission.py      # Prefill decoder layer kernel
│   ├── baseline.py        # SGLang decoder layer baseline
│   ├── reference.py       # Prefill correctness reference
│   ├── TASK.md            # Prefill problem spec
│   └── profiles/          # Remote H200 prefill profile breakdown
├── muse_bf16_proj/        # Dense projections problem workspace
├── muse_attention_decode/ # Gated attention decode problem workspace
└── serving/
    └── RULES.md           # End-to-end SGLang serving challenge rules
```

---

## Author & Credits

- **Team:** `ruwwww`
- **Committer:** `kobeniw` (`kobeniw@users.noreply.github.com`)
- **Challenge:** [Open Kernel Challenge](https://openkernelchallenge.dev/) (Meta & Andromeda)
