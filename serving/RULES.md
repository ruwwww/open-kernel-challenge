# The serving track

Make SGLang serve Muse Spark Open faster. Your submission is a patch to the SGLang checkout in
`sglang/`, at commit `a55ab7d5`, the one LMSYS's image `lmsysorg/sglang:dev-muse-spark-open`
serves the model with. Edit it, then:

    okc serve diff                     # writes patch.diff
    okc serve check patch.diff         # a short unscored run: correctness and one load
    okc serve submit patch.diff        # a scored run

A scored run takes a whole 8×H200 node for up to 90 minutes. It serves multi-turn agentic
coding conversations at concurrency 1, 4, 16 and 32, and scores your throughput at stock
SGLang's interactivity on the same node. Every request must return exactly 220 tokens, your
answers must match stock's, and your time to first token can be at most 2× stock's at each
load. You get 1 scored run and 3 check runs a day.

## Running it yourself

On an 8-GPU Hopper node, in the image, with your checkout installed in place of the image's
SGLang (`pip install -e sglang/python --no-deps`):

    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=1800 python -m sglang.launch_server \
      --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
      --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 --tp-size 8 --ep-size 8 \
      --quantization modelopt_fp4 --moe-runner-backend marlin --mem-fraction-static 0.85 \
      --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
      --enable-hierarchical-cache --chunked-prefill-size 16384 --context-length 131072 \
      --enable-cache-report --incremental-streaming-output --random-seed 1234 \
      --max-running-requests 32 --cuda-graph-max-bs-decode 32

These are the judge's flags. Your patch can change what they do, not the flags themselves.
To load it, `python -m sglang.bench_serving --backend sglang --max-concurrency 16` is a
start; the judge's conversations are its own and stay hidden.

## Where to start

All under `sglang/python/sglang/srt/`:

- `models/muse_spark.py`: the model: its layers, attention, MoE block and norms.
- `layers/moe/moe_runner/marlin.py`, `layers/moe/fused_moe_triton/fused_marlin_moe.py`: the
  routed experts on Hopper (Marlin W4A16), the biggest share of a decode step from 16 requests up.
- `layers/attention/flashattention_backend.py`: attention over the FP8 KV cache (FA3).
- `speculative/dflash_worker_v2.py`, `speculative/dflash_utils.py`: DFlash speculative
  decoding, 8 draft tokens.
- `managers/scheduler.py`, `managers/schedule_policy.py`: batching and scheduling.
- `model_executor/model_runner.py`, `model_executor/cuda_graph_config.py`: CUDA graphs.

Faster kernels from the kernel problems plug in at the same places.

## Patch rules

- a unified diff (git diff) against the base commit, changing only files under python/sglang/
- not weight loading (python/sglang/srt/model_loader/)
- only source files (.py, .cu, .cuh, .h, .hpp, .c, .cc, .cpp, .json); no renames, mode changes or binary patches
- no added line over 400 characters; at most 720 KB
- the model must stay the model: 8 routed experts per token, the 2,048-token window, every norm
- nothing below stock's precision: activations stay BF16 wherever stock keeps them in BF16 (stock's FP8 query in FA3 attention stays as it is), and the FP8 KV cache stays FP8
- the drafter is DFlash with its released weights: make it run faster, but don't change, retrain or replace it
- other drafting may only use the requests themselves: prompt lookup and n-grams from the conversation

The rules: https://openkernelchallenge.dev/rules
