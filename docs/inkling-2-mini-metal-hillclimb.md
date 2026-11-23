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

One session on the night of 22–23 Nov 2026. The three setups ran in the order A B C C B A, three times over, so each
ran six times. Settings: llama-server, 131k context, 2 slots, flash attention on. Each run takes the median of 6
decode samples.

Speeds are the mean of the six runs ± their standard deviation, the convention llama-bench uses. With six runs,
mean ± sd is also very close to a 95% confidence interval for the mean.

| t/s, n = 6 | stock `418fe97` + Q8_0 | this branch, exact (default) | this branch, fast (opt-in) |
|---|---|---|---|
| decode, 1 user | 28.6 ± 0.1 | **47.7 ± 0.9** | 48.0 ± 0.3 |
| decode, 2 users (total) | 40.0 ± 0.3 | **65.4 ± 2.5** | 65.1 ± 2.3 |
| prefill, 3.3k-token prompt | 382 ± 5 | **1021 ± 60** | 1062 ± 37 |
| prefill, 12.9k-token prompt | 309 ± 3 | **998 ± 28** | 1039 ± 32 |

Gains, with 95% confidence intervals:

| | exact vs stock | fast vs stock | fast vs exact |
|---|---|---|---|
| decode, 1 user | **+67%** (+64 to +70%) | +68% (+67 to +69%) | +1% (−1 to +3%) |
| decode, 2 users | **+63%** (+57 to +70%) | +63% (+57 to +69%) | 0% (−5 to +4%) |
| prefill, 3.3k-token prompt | **×2.67** (2.52–2.84) | ×2.78 (2.68–2.89) | +4% (−2 to +11%) |
| prefill, 12.9k-token prompt | **×3.23** (3.14–3.32) | ×3.37 (3.26–3.48) | +4% (0 to +8%) |

The intervals are computed on the log of the ratio of means, from each side's standard error with Welch's degrees of
freedom.

- All runs come from one evening on one Mac. The intervals cover run-to-run noise only, not differences between
  machines or days.
- The runs drifted up as the machine got quieter. Exact prefill on the 3.3k-token prompt went from 900 t/s on the
  first run to about 1050, and stock rose about 3%. So the runs are not fully independent.
- Fast against exact: decode is the same, and prefill is about 4% higher, at the edge of the noise.

<details><summary>Raw values (run order)</summary>

| t/s | stock + Q8_0 | exact | fast |
|---|---|---|---|
| decode, 1 user | 28.4 / 28.6 / 28.6 / 28.6 / 28.7 / 28.4 | 45.8 / 48.2 / 48.3 / 48.3 / 47.6 / 48.0 | 48.0 / 48.3 / 48.1 / 47.6 / 47.9 / 48.3 |
| decode, 2 users (total) | 39.4 / 40.1 / 40.0 / 40.2 / 40.2 / 40.2 | 60.6 / 66.7 / 66.8 / 64.8 / 67.3 / 65.9 | 60.5 / 66.2 / 65.1 / 66.1 / 66.1 / 66.6 |
| prefill, 3.3k-token prompt | 372 / 383 / 384 / 383 / 384 / 385 | 900 / 1041 / 1025 / 1046 / 1062 / 1049 | 997 / 1039 / 1093 / 1091 / 1073 / 1083 |
| prefill, 12.9k-token prompt | 303 / 308 / 311 / 310 / 310 / 311 | 953 / 977 / 1003 / 1006 / 1021 / 1026 | 990 / 1010 / 1049 / 1059 / 1053 / 1075 |

</details>

Other measurements of the exact build:

- llama-bench pp512 is 1183 t/s (separate run, different harness).
- A 119k-token needle prompt (separate run) prefills at 544 t/s, against 174 t/s for the day-1 build.

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

Steps 3–6 keep the model's outputs bit-identical. Single-user decode stayed flat after step 2: it was already near
its limit (see Limits).

## Exactness

The changes from step 3 onwards keep results bit-identical to the build before them. Each one was checked with all
of these:

- a wikitext KLD run identical to every printed digit;
- greedy panels with top-5 logprobs;
- a fixed 20.8k-token greedy run;
- perplexity at 2-, 3-, 4- and 512-token batches;
- `test-backend-ops`.

Some day-1 fusions change rounding. Against the Q8_0 logits on wikitext, KLD is 0.0015 on the prefill path and
0.0017 on the decode path. That is the floating-point noise floor for this model, which turns last-bit rounding
differences into visible greedy divergence.

The fast column is opt-in and changes rounding: `-ub 1024 --cache-ram 4096` plus `GGML_METAL_MMID_MIN_T=6`.

- KLD against the exact path is 0.0002.
- Prefill: +4% (95% CI 0 to +8%) at 12.9k tokens in the table above, and +7% in llama-bench pp4096.
- Follow-up prompts of 6–31 tokens run 12–64% faster.
- GPU peak is 46 GiB instead of 35 GiB.

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
- Tensor-API matmuls with few output rows use 32-token tiles instead of 128. Set `GGML_METAL_MM_NARROW_DISABLE` to
  turn it off.
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
- The fast setup needs about 11 GiB more GPU memory. With other apps open it caused 140–470 MiB of swap in 3 of its 6
  runs above; the exact build caused none.
- The relative-position bias inside flash attention (upstream PR 32740 plus a Metal port) lives in a separate local
  branch. It saves about 11 GiB of compute buffer but is slower at prefill than this branch's mask kernel.
- The DSpark drafter is still 12–40% slower than plain decoding on this machine.
