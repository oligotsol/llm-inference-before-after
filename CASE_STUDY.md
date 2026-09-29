# Case Study: Before & After LLM Inference Optimization

**Repository:** https://github.com/oligotsol/llm-inference-before-after  
**Author:** oli (oligotsol)  
**Measured:** 2026-09-28 (America/Chicago)  
**Stack:** llama-cpp-python 0.3.16 · llama.cpp GGML CPU · Qwen2.5-0.5B-Instruct GGUF (bartowski)

---

## Executive summary

We replicated a popular open-source local inference pipeline (**llama.cpp via llama-cpp-python**) and applied **GGUF weight quantization** (F16 → Q4_K_M) on a CPU-only box (~15 GiB RAM, no CUDA).

| Metric | BEFORE (F16) | AFTER (Q4_K_M) | Improvement |
| --- | ---: | ---: | ---: |
| Mean end-to-end latency | 5.343 s | 2.229 s | **58.3% faster** |
| Mean generation throughput | 26.17 tok/s | 57.63 tok/s | **+120.2%** |
| Mean peak RSS | 2006.6 MB | 722.2 MB | **64.0% less memory** |
| On-disk GGUF size | 948.1 MB | 379.4 MB | −60% |

**Success criteria:** ≥50% latency reduction **or** ~half memory (or better).  
**Result:** **Both** hit — 58.3% latency drop and 64.0% peak RSS drop.

All numbers below come from committed JSON under `results/` produced by `scripts/bench.py` on this machine. Nothing was invented.

---

## Pipeline

```
Fixed prompts  →  llama-cpp-python Llama.create_chat_completion
                       │
                       ├─ BEFORE: Qwen2.5-0.5B-Instruct-f16.gguf
                       ├─ MID:    Qwen2.5-0.5B-Instruct-Q8_0.gguf
                       └─ AFTER:  Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
                       │
                 warmup excluded → N timed runs → mean latency / tok/s / peak RSS
```

Why this stack:

- **llama.cpp / GGUF** is the de-facto local CPU inference path used by Ollama, LM Studio, and many tutorials.
- **0.5B Instruct** fits comfortably in 15 GiB even at F16, so BEFORE and AFTER share identical settings (threads, ctx, max tokens).
- Quantized weights ship as ready-made GGUFs from [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF) — no custom quantizer required.

---

## Techniques applied

### 1. Weight quantization (primary)

| Variant | Role | Bits (approx.) | File size |
| --- | --- | --- | ---: |
| F16 | BEFORE baseline | 16-bit floats | 948.1 MB |
| Q8_0 | Mid-point | ~8-bit | 506.5 MB |
| Q4_K_M | AFTER | ~4.8-bit K-quant | 379.4 MB |

Q4_K_M (“K-quant medium”) is llama.cpp’s recommended default: mixed block sizes with higher precision on important tensors. On CPU, smaller weights mean:

1. Less DRAM bandwidth per matmul (often the bottleneck).
2. Better cache residency.
3. Smaller peak resident set for the same context window.

### 2. Prompt-lookup speculative decoding (explored)

llama-cpp-python exposes `LlamaPromptLookupDecoding` — an n-gram draft mechanism (no second model). We ran Q4_K_M + PLD (`ngram=3`, `pred=8`).

On this **short, low-repetition** instruct workload, PLD was **slower** than plain Q4 (mean 4.121 s vs 2.229 s). Draft overhead exceeded accept benefit. We document it as an honest negative result rather than forcing a “win.”

A separate tiny draft GGUF was not practical: 0.5B is already near the bottom of the Qwen2.5 Instruct family.

---

## Hardware under test

Captured automatically by `scripts/bench.py` (also embedded in each results JSON):

| Field | Value |
| --- | --- |
| OS / kernel | `Linux grok-bot-vm-413818549 6.12.94+ #1 SMP PREEMPT_DYNAMIC Thu Sep 24 16:04:37 UTC 2026 x86_64 GNU/Linux` |
| CPU model | Intel(R) Xeon(R) Processor |
| Logical CPUs (`nproc`) | 8 |
| RAM total | 16013 MB (~15 GiB) |
| Python | 3.13.5 |
| GPU / CUDA | **None** (CPU-only box) |
| Threads used | 8 |
| Context (`n_ctx`) | 2048 |

`free -h` snapshot at BEFORE run:

```
total        used        free      shared  buff/cache   available
Mem:            15Gi       9.9Gi       1.2Gi        45Mi       4.9Gi       5.8Gi
Swap:             0B          0B          0B
```

---

## Measurement protocol

| Setting | Value |
| --- | --- |
| Prompts | 3 fixed technical prompts (see `scripts/bench.py`) |
| `max_tokens` | 128 |
| Temperature | 0.0 (deterministic) |
| Warmup | 1 per prompt (**excluded** from stats) |
| Timed runs | 3 per prompt → **9** timed samples per config |
| Latency | Wall time around `create_chat_completion` |
| Throughput | `completion_tokens / elapsed_s` |
| Peak memory | Background RSS sampler (`psutil`, 50 ms) during generation |
| Mean / stdev | Across all timed runs |

Prompts (verbatim):

1. *Explain how quantization reduces memory usage in large language models in two short paragraphs.*
2. *Write a Python function that computes Fibonacci numbers iteratively, with a brief docstring.*
3. *List five practical tips for optimizing CPU inference throughput with llama.cpp.*

---

## Results

### Headline table (committed runs)

| Config | Label | Mean latency (s) | Stdev (s) | Mean tok/s | Peak RSS (MB) | File (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BEFORE F16 | `before_f16` | 5.343 | 2.420 | 26.17 | 2006.6 | 948.1 |
| MID Q8_0 | `mid_q8_0` | 2.118 | 0.338 | 57.37 | 1182.8 | 506.5 |
| AFTER Q4_K_M | `after_q4_k_m` | 2.229 | 0.142 | 57.63 | 722.2 | 379.4 |
| AFTER Q4 + PLD | `after_q4_k_m_pld` | 4.121 | 0.453 | 30.53 | 870.2 | 379.4 |

### Before → After deltas (F16 → Q4_K_M)

- Latency: **58.3%** reduction (5.343 → 2.229 s)
- Peak RSS: **64.0%** reduction (2006.6 → 722.2 MB)
- Throughput: **+120.2%** (26.17 → 57.63 tok/s)

### Observations

1. **Q4_K_M dominates F16** on both latency and memory — the clear production choice for CPU serving of this model size.
2. **Q8_0** lands between F16 and Q4 on memory (1183 MB) with throughput similar to Q4 on this host (~57 tok/s). Prefer Q8 when quality sensitivity is high and RAM allows.
3. **PLD speculative decoding** did not help here; keep it for workloads with repeated spans (RAG with copied passages, code with boilerplate).
4. Peak RSS ≫ file size for all configs because of Python runtime, KV cache (`n_ctx=2048`), and llama.cpp scratch buffers — compare **like-for-like** RSS, not file size alone.

Raw per-run samples: `results/before_f16.json`, `results/after_q4_k_m.json`, `results/mid_q8_0.json`, `results/after_q4_k_m_pld.json`, plus rollup `results/comparison.json`.

---

## How to reproduce

```bash
git clone https://github.com/oligotsol/llm-inference-before-after.git
cd llm-inference-before-after
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ~1.8 GB download
python scripts/download_models.py --dir ./models   # or /workspace/models

python scripts/bench.py \
  --model ./models/Qwen2.5-0.5B-Instruct-f16.gguf \
  --label before_f16 --max-tokens 128 --runs 3 --warmup 1 --threads $(nproc) \
  --out results/before_f16.json

python scripts/bench.py \
  --model ./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --label after_q4_k_m --max-tokens 128 --runs 3 --warmup 1 --threads $(nproc) \
  --out results/after_q4_k_m.json

# Optional mid-point and PLD
python scripts/bench.py --model ./models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --label mid_q8_0 --max-tokens 128 --runs 3 --warmup 1 --threads $(nproc) \
  --out results/mid_q8_0.json

python scripts/bench.py --model ./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --label after_q4_k_m_pld --prompt-lookup --max-tokens 128 --runs 3 --warmup 1 \
  --threads $(nproc) --out results/after_q4_k_m_pld.json

python scripts/compare.py
```

**Build notes:** `llama-cpp-python` compiles llama.cpp from source. Needs `g++`, `cmake`, and (recommended) `pkg-config`. No CUDA required for this study.

Absolute numbers will differ on other CPUs, but the **relative** F16 → Q4_K_M gap should remain large on memory-bandwidth-bound machines.

---

## Caveats & fairness

- Same prompts, `max_tokens`, threads, and context for every config.
- Warmup excluded so cold-start / page-cache effects do not dominate means.
- Output length can vary slightly when the model emits EOS early (F16 sometimes stopped before 128 tokens on prompt 1); tokens/sec accounts for that.
- Quality was not scored with an automatic eval harness; Q4_K_M is widely considered the quality/size sweet spot for llama.cpp. Spot-check responses in the JSON `response_preview` fields.
- This is **CPU** inference. GPU FP16 vs INT4/AWQ would use a different stack (vLLM, TensorRT-LLM, etc.).

---

## Conclusion

Quantizing Qwen2.5-0.5B-Instruct from **F16 → Q4_K_M** in llama.cpp delivered a **58.3% latency cut** and **64.0% peak memory cut** under a fixed, reproducible protocol — clearing the ≥50% latency bar and the ~half-memory bar simultaneously. Speculative prompt-lookup decoding was measured and did not improve this workload; the winning lever was quantization.

---

## Appendix: result file timestamps (local)

| File | `timestamp_local` |
| --- | --- |
| before_f16 | 2026-09-28T21:52:58.166720-05:00 |
| mid_q8_0 | 2026-09-28T21:56:07.621252-05:00 |
| after_q4_k_m | 2026-09-28T21:53:30.637687-05:00 |
| after_q4_k_m_pld | 2026-09-28T21:55:32.076444-05:00 |
