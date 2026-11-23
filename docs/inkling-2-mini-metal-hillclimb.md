# Inkling 2 Mini on Apple Silicon: a Metal speed hill climb

This fork branch (`inkling-hc`) makes [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) run
faster on llama.cpp's Metal backend. The work was done on an Apple M5 Pro with 64 GB of unified memory, over
21–22 Nov 2026, starting from upstream master `418fe97`. It is not submitted upstream.

The model is a looped MoE. Its 24 layers run as 48 passes, with 128 routed experts (8 active) and a learned
relative-position attention bias. In stock llama.cpp that means many small kernels per token and a large attention
mask built on every full-attention layer. Most of the work below goes after those two costs.

The method follows [0xSero](https://github.com/0xSero)'s
[inference-optimisation agent skill](https://github.com/sybil-solutions/inference-optimisation-skill):

- compute the theoretical ceiling first;
- change one thing at a time;
- A/B every change against controls run before and after it;
- keep or reject it, and log why;
- put the winner through long-context and concurrency checks.

The work was AI-assisted (Claude Code), and every change was measured on this machine.

## Results

One session on the morning of 23 Nov 2026, on a quiet machine. Setups ran in the order A B C D D C B A, three times
over, so each ran six times. Settings: llama-server, 131k context, 2 slots, flash attention on. Each run takes the
median of 6 decode samples. The branch columns used `213ae99`; the one later change (`dc7339e`) only affects 6–31-token
batches (below). The previous published numbers (`703cd33`) are in the second column for comparison.

Speeds are the mean of the six runs ± their standard deviation, the convention llama-bench uses. With six runs,
mean ± sd is also very close to a 95% confidence interval for the mean.

| t/s, n = 6 | stock `418fe97` + Q8_0 | branch at `703cd33` | this branch, exact (default) | this branch, fast (opt-in) |
|---|---|---|---|---|
| decode, 1 user | 29.1 ± 0.0 | 49.4 ± 0.0 | **49.2 ± 0.1** | 49.2 ± 0.1 |
| decode, 2 users (total) | 40.7 ± 0.1 | 67.6 ± 0.1 | **67.7 ± 0.3** | 67.7 ± 0.2 |
| prefill, 3.3k-token prompt | 386 ± 2 | 1065 ± 1 | **1183 ± 2** | 1238 ± 16 |
| prefill, 12.9k-token prompt | 312 ± 0 | 1027 ± 1 | **1163 ± 1** | 1232 ± 2 |

Gains, with 95% confidence intervals:

| | exact vs stock | exact vs `703cd33` | fast vs exact |
|---|---|---|---|
| decode, 1 user | **+69%** (+69 to +70%) | 0% (see below) | 0% (−0 to +0%) |
| decode, 2 users | **+66%** (+66 to +67%) | 0% (−0 to +1%) | 0% (−1 to +0%) |
| prefill, 3.3k-token prompt | **×3.07** (3.05–3.08) | +11% (+11 to +11%) | +5% (+3 to +6%) |
| prefill, 12.9k-token prompt | **×3.73** (3.72–3.73) | +13% (+13 to +13%) | +6% (+6 to +6%) |

The intervals are computed on the log of the ratio of means, from each side's standard error with Welch's degrees of
freedom.

- All runs come from one session on one Mac. The intervals cover run-to-run noise only, not differences between
  machines or days.
- Single-user decode reads 0.4% lower than `703cd33` here, but none of the later changes touches the one-token path:
  llama-bench decode is identical, and server-level decode moves by up to ±0.5% between builds in no particular
  direction (code generation and memory layout).
- The fast setup swapped 2.4 GB in one of its six runs (46 GiB of GPU memory with other apps open).

<details><summary>Raw values (run order)</summary>

| t/s | stock + Q8_0 | exact | fast |
|---|---|---|---|
| decode, 1 user | 29.1 / 29.0 / 29.1 / 29.1 / 29.1 / 29.1 | 49.1 / 49.2 / 49.3 / 49.3 / 49.3 / 49.2 | 49.1 / 49.3 / 49.2 / 49.1 / 49.3 / 49.3 |
| decode, 2 users (total) | 40.7 / 40.9 / 40.7 / 40.7 / 40.8 / 40.7 | 67.9 / 67.2 / 67.9 / 67.7 / 68.0 / 67.7 | 67.3 / 67.8 / 67.9 / 67.8 / 67.7 / 67.7 |
| prefill, 3.3k-token prompt | 382 / 387 / 387 / 386 / 387 / 386 | 1184 / 1178 / 1182 / 1184 / 1184 / 1184 | 1241 / 1250 / 1233 / 1258 / 1213 / 1234 |
| prefill, 12.9k-token prompt | 311 / 312 / 312 / 312 / 313 / 312 | 1164 / 1163 / 1165 / 1163 / 1162 / 1163 | 1227 / 1232 / 1232 / 1232 / 1233 / 1234 |

</details>

Other measurements of the exact build (separate runs):

- llama-bench pp512: 1289 t/s; at 64k of context about 570 t/s.
- A 119k-token needle prompt prefills at 624 t/s, against 544 for `703cd33` and 174 for the day-1 build.
- Short batches at 4k of context (`dc7339e`): 16 tokens 157.5 t/s, 24 tokens 153 t/s (+8% over `213ae99`).

These numbers don't include speculative decoding. With `--spec-type ngram-mod` (lossless), file edits and rewrites
run at 78–137 t/s.

Both branch columns use the ggml-org Q8_0 with the routed experts of layers 0–19 replaced by the released NVFP4
tensors, taken unchanged from Unsloth's UD-Q8_K_XL. Layers 20–23 keep Q8_0, because their originals are MXFP8.
That file needs the NVFP4 `MUL_MAT_ID` kernels from this branch. The plain Q8_0 also works.

### Where the gains came from

Each step was measured against the build before it, in the same session:

| Step | What changed | Effect |
|---|---|---|
| 1. Day 1, kernels only, Q8_0 | Graph dedup across loop passes, 12 Metal fusions, packed projections, wider reorder | decode 29.2 → 39.1 t/s; prefill 387 → 562 (4k), 312 → 459 (16k) |
| 2. NVFP4 experts | Released NVFP4 expert tensors, Metal NVFP4 `MUL_MAT_ID`, rel-pos mask from a bias table, one-token fusions | decode → 49.2; 2 users → 64.0; prefill → 863 (4k), 830 (16k) |
| 3. Prompt batches | Depth-mix tail for token batches, faster rel-pos mask kernel | prefill +11% (4k), +15% (16k); 2 users +3%; llama-bench pp512 929 → 1112 |
| 4. Two users | Fusions also fire for 2 sequences and for small batches | 2 users +2.8% (65.4 → 67.2) |
| 5. Short conv | Removed a 64-bit index division per thread in the prompt-batch short conv | prefill +3.3% (1006 → 1039 at 4k) |
| 6. Narrow tiles | 32-token tiles for tensor-API matmuls with few output rows | llama-bench pp512 1162 → 1183 (+1.8%) |
| 7. K steps | Few-row matmuls (sinks, loop control, router) take K in steps of 128 | +0.5% prefill |
| 8. Expert tiles 64 | Expert matmul with 64-entry tiles: weights read once per 64 tokens | +1.8% prefill |
| 9. FA tiles | Flash attention with 16 queries per threadgroup while KV ≤ 24k (more threadgroups per core) | +3.5–4% at 4k–16k |
| 10. Rel-pos mask | Mask kernel with 2 heads per threadgroup (8 KB table, 3 threadgroups per core) | +2.8% (+4.2% at 4k) |
| 11. Block scan | Flash-attention block scan on the shared base mask instead of every head's mask | +0.8% at 4k, +5% at 64k |
| 12. Expert rows 128 | Expert matmul with 128 weight rows per threadgroup (8 simdgroups) | +1.3% prefill |
| 13. Short batches | Expert matvec for 6–31 tokens: each expert's tokens in one threadgroup, weights from cache | +4–8.5% on 6–31-token batches |

Steps 3–13 keep the model's outputs bit-identical. Single-user decode stayed flat after step 2: it was already near
its limit (see Limits).

## Exactness

The changes from step 3 onwards keep results bit-identical to the build before them. Each one was checked with all
of these:

- a wikitext KLD run identical to every printed digit;
- greedy panels with top-5 logprobs;
- a fixed 20.8k-token greedy run;
- perplexity at 2-, 3-, 4-, 8-, 16-, 24- and 512-token batches;
- for the changes that touch long context or small batches, perplexity at 8k and 32k of context and at 8-, 16-
  and 24-token batches with the change on and off, identical per chunk;
- `test-backend-ops`.

Some day-1 fusions change rounding. Against the Q8_0 logits on wikitext, KLD is 0.0015 on the prefill path and
0.0017 on the decode path. That is the floating-point noise floor for this model, which turns last-bit rounding
differences into visible greedy divergence.

The fast column is opt-in and changes rounding: `-ub 1024 --cache-ram 4096` plus `GGML_METAL_MMID_MIN_T=6`.

- KLD against the exact path is 0.0002.
- Prefill: +5% at 3.3k tokens and +6% at 12.9k in the table above (95% CI +3 to +6% and +6 to +6%).
- Follow-up prompts of 6–31 tokens run 12–64% faster.
- GPU memory is about 46 GiB instead of 34 GiB.

Qualified at `dc7339e`: a 119k-token needle found 3/3 (prefill 624 t/s); two concurrent ~55k-token needles found
6/6; GPU peak 33 GiB; no swap. The full `test-backend-ops` suite on Metal passes (19,782 of 19,782).

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
- Tensor-API matmuls with few output rows use 32-token tiles instead of 128. Set `GGML_METAL_MM_NARROW_DISABLE` to
  turn it off. Those tiles take K in steps of 128 instead of 32 (`GGML_METAL_MM_NARROW_NK=2` restores 32).
- Tensor-API `mul_mm_id` (expert matmul) uses 64-entry tiles, so each expert's weights are read once per 64 tokens,
  and 128 weight rows per threadgroup with 8 simdgroups (`GGML_METAL_MMID_NR64_DISABLE`,
  `GGML_METAL_MMID_R128_DISABLE`).
- Tensor flash attention takes 16 queries per threadgroup for f16 with dk = dv = 128 while the KV is at most 24k
  long, and keeps the 32-query tile past that (`GGML_METAL_FA_TENSOR_Q32` forces 32).
- The flash-attention block scan reads the shared F32 base mask once instead of every head's mask, when the mask is
  built from a bias table (`GGML_METAL_FA_BLK_BM_DISABLE`).
- The rel-pos mask kernel handles 2 heads per threadgroup for prompt batches (8 KB table, 3 threadgroups per core;
  `GGML_METAL_RELPOS_HG=4` restores 4).
- For 6–31-token batches, the NVFP4 expert matvec runs all of an expert's tokens in one threadgroup with the
  unchanged single-token code, so the weights come from cache after the first token (`GGML_METAL_MVID_GROUP_DISABLE`).
- All of these keep outputs bit-identical; each has the switch shown to turn it off.
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

For the fast setup, add `-ub 1024 --cache-ram 4096` and set `GGML_METAL_MMID_MIN_T=6`.

## Limits

- Everything was measured on one M5 Pro, with measured memory bandwidth of 297 GB/s. Other Apple GPUs are untested.
  The graph-level changes also affect other backends, but they were not tested there.
- Single-user decode is close to its practical limit for exact kernels. Each token still runs about 1,750 dependent
  kernels at roughly 4–5 µs each, and the weight kernels already run near bandwidth.
- The fast setup needs about 12 GiB more GPU memory. With other apps open it swapped 2.4 GB in one of its six runs
  above; the exact build caused none.
- The relative-position bias inside flash attention (upstream PR 32740 plus a Metal port) lives in a separate local
  branch. It saves about 11 GiB of compute buffer but is slower at prefill than this branch's mask kernel.
- The DSpark drafter is still 12–40% slower than plain decoding on this machine.
