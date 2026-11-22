# Inkling 2 Mini on Apple Silicon: a Metal speed hill climb

This fork branch (`inkling-hc`) makes [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) run
faster on llama.cpp's Metal backend. The work was done on an Apple M5 Pro with 64 GB of unified memory, over
21–22 Nov 2026, starting from upstream master `418fe97`. It is not submitted upstream.

The model is a looped MoE. Its 24 layers run as 48 passes, with 128 routed experts (8 active) and a learned
relative-position attention bias. In stock llama.cpp that means many small kernels per token and a large attention
mask built on every full-attention layer. Most of the work below goes after those two costs.

The method: measure, change one thing, A/B against controls on both sides, keep it or reject it, and log why. The
work was AI-assisted (Claude Code), and every change was measured on this machine.

## Results

Same session, interleaved runs, medians (llama-server, 131k context, 2 slots, flash attention on):

| | stock `418fe97` + Q8_0 | day 1, kernels only, + Q8_0 | this branch + NVFP4 experts |
|---|---|---|---|
| decode, 1 user | 29.2 t/s | 39.1 | **49.2** (+69%) |
| decode, 2 users (total) | 40.9 | 50.1 | **64.0** (+56%) |
| prefill, ~4k tokens | 387 t/s | 562 | **863** (+123%) |
| prefill, ~16k tokens | 312 t/s | 459 | **830** (+166%) |

The model's outputs did not change in any of the later work:

- prefill went up about 25% more (llama-bench pp512: 929 → 1183);
- 2-user decode went up about 5% (67.2 t/s total);
- 1-user decode stayed the same.

These numbers don't include speculative decoding. With `--spec-type ngram-mod` (lossless), file edits and rewrites
run at 78–137 t/s.

The model file in the last column is the ggml-org Q8_0 with the routed experts of layers 0–19 replaced by the
released NVFP4 tensors, taken unchanged from Unsloth's UD-Q8_K_XL. Layers 20–23 keep Q8_0, because their
originals are MXFP8. That file needs the NVFP4 `MUL_MAT_ID` kernels from this branch. The plain Q8_0 also works.

## Exactness

The most recent changes keep results bit-identical to the build before them. Each one was checked against the
previous build with all of these:

- a wikitext KLD run identical to every printed digit;
- greedy panels with top-5 logprobs;
- a fixed 20.8k-token greedy run;
- perplexity at 2-, 3-, 4- and 512-token batches;
- `test-backend-ops`.

Some day-1 fusions change rounding. Against the Q8_0 logits on wikitext, KLD is 0.0015 on the prefill path and
0.0017 on the decode path. That is the floating-point noise floor for this model, which turns last-bit rounding
differences into visible greedy divergence.

Two opt-ins that change rounding stay off by default:

- `GGML_METAL_MMID_MIN_T=6` makes 6–31-token batches 12–64% faster;
- `-ub 1024` adds about 7% prefill at 4k–16k context.

Qualified: a 119k-token needle found 3/3; two concurrent ~55k-token needles found 6/6; GPU peak 35 GiB; no swap.

## What is in the branch

**Graph (`src/models/inkling2*.cpp`)**
- Normed block stacks, message norms and weight-only ops are built once per graph instead of once per loop pass.
  Graph nodes −9%, decode +2.5–4.2%, outputs identical.
- Projections are packed after load (Q, gate, rel and sink; loop-control gain|forget; K/V rows with the attention
  projections), so each takes one matvec at one token. Set `LLAMA_INKLING_NO_PACK=1` to turn it off.

**NVFP4**
- Metal `MUL_MAT_ID` kernels for NVFP4 (matvec and mm_id). The per-expert scale is folded into the matvec for
  decode and into the mm_id store for prefill.
- A fix for loop slots that never got the per-expert scales, which gave perplexity 121 instead of 7.09. Upstream has
  since changed how weight scales are looked up (PR 32669, by tensor), which should cover this. That has not been
  re-tested.

**Attention**
- The relative-position mask is built from a bias table kept in threadgroup memory, instead of one dot product per
  mask element. Prefill +44–72%, ×2.5 at 64k.

**Metal fusions, each one exact or checked against its reference**
- SITU GLU.
- Row L2 clip.
- Short convolution: plain and dynamic-tap, for decode, prompt batches and several sequences.
- Loop-control blend.
- Depth-mix tail.
- Attention output gate folded into the `wo` matvec.
- Concat + matvec.
- K/V cache stores folded into the clip and norm kernels.

Most of these have a `GGML_METAL_<NAME>_DISABLE` switch and a `test-backend-ops` case.

**Generic Metal**
- The graph reorder can move across plain CONCAT, SQR, SQRT, FILL, PAD and SOFT_MAX nodes, with a bounded
  lookahead so time to first token doesn't grow.
- Tensor-API matmuls with few output rows use 32-token tiles instead of 128 (prefill +1.7%). Set
  `GGML_METAL_MM_NARROW_DISABLE` to turn it off.
- Profiling switches:
  - `GGML_METAL_OPPROF=1`: GPU time per op group;
  - `GGML_METAL_OP_TWICE=<pattern>`: encodes matching groups twice, to measure their cost inside the pipeline;
  - `GGML_METAL_TIMING=1`: per-graph encode and wait times.

## Run it

```bash
cmake -B build && cmake --build build --config Release -j
```

```bash
build/bin/llama-server -m Inkling-2-Mini-Q8_0-nvfp4exp.gguf -a inkling-2-mini \
  -c 131072 -np 2 --kv-unified -fa on --jinja -ub 512 --cache-ram 8192 -ngl 99 -fit off \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0
```

## Limits

- Everything was measured on one M5 Pro, with measured memory bandwidth of 297 GB/s. Other Apple GPUs are untested.
  The graph-level changes also affect other backends, but they were not tested there.
- Single-user decode is close to its practical limit for exact kernels. Each token still runs about 1,750 dependent
  kernels at roughly 4–5 µs each, and the weight kernels already run near bandwidth.
- The relative-position bias inside flash attention (upstream PR 32740 plus a Metal port) lives in a separate local
  branch. It saves about 11 GiB of compute buffer but is slower at prefill than this branch's mask kernel.
- The DSpark drafter is still 12–40% slower than plain decoding on this machine.
