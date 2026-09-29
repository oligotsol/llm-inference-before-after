# LLM Inference Before & After

**Real measured case study:** cut local LLM inference latency and memory with GGUF quantization on CPU — no invented metrics.

| | BEFORE (F16) | AFTER (Q4_K_M) | Δ |
| --- | ---: | ---: | ---: |
| **Mean latency** (128 tok gen) | **5.343 s** | **2.229 s** | **−58.3%** |
| **Mean throughput** | 26.17 tok/s | 57.63 tok/s | **+120%** |
| **Peak RSS** | 2006.6 MB | 722.2 MB | **−64.0%** |
| Model file | 948 MB | 379 MB | −60% |

**Goal met:** ≥50% latency drop **and** better-than-half peak memory on the same prompts, hardware, and token budget.

Stack: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.16 (CPU llama.cpp) · model: [Qwen2.5-0.5B-Instruct GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF) (bartowski) · hardware: 8× Intel Xeon, 15 GiB RAM, no GPU.

Full narrative, tables, and reproduce steps → **[CASE_STUDY.md](./CASE_STUDY.md)** · raw JSON → [`results/`](./results/).

---

## Quick start

```bash
git clone https://github.com/oligotsol/llm-inference-before-after.git
cd llm-inference-before-after

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads ~1.8 GB of GGUF weights (gitignored)
python scripts/download_models.py --dir /workspace/models   # or ./models

# BEFORE (F16)
python scripts/bench.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct-f16.gguf \
  --label before_f16 --max-tokens 128 --runs 3 --warmup 1 --threads 8 \
  --out results/before_f16.json

# AFTER (Q4_K_M)
python scripts/bench.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --label after_q4_k_m --max-tokens 128 --runs 3 --warmup 1 --threads 8 \
  --out results/after_q4_k_m.json

python scripts/compare.py
```

Optional mid-point (`Q8_0`) and prompt-lookup speculative decoding (`--prompt-lookup`) are documented in the case study.

---

## What’s in the repo

| Path | Purpose |
| --- | --- |
| `scripts/bench.py` | Fixed prompts, warmup excluded, N timed runs, peak RSS sampling |
| `scripts/download_models.py` | Pull F16 / Q8_0 / Q4_K_M GGUF from Hugging Face |
| `scripts/compare.py` | Print markdown comparison from `results/*.json` |
| `results/*.json` | **Committed** measurements from this machine |
| `CASE_STUDY.md` | Full write-up: pipeline, techniques, hardware, caveats |
| `requirements.txt` | Pinned deps |

Models stay under `/workspace/models` (or `./models`) and are **gitignored**.

---

## Techniques

1. **Weight quantization (primary win):** F16 → Q4_K_M GGUF via llama.cpp kernels — less memory traffic, faster CPU matmuls, ~half the resident set.
2. **Q8_0 mid-point:** near-F16 quality with much less weight memory; on this box, throughput close to Q4 but ~1.6× the Q4 peak RSS.
3. **Prompt-lookup decoding (explored):** n-gram speculative drafts via `LlamaPromptLookupDecoding`. On this short, low-repetition workload it **did not** beat plain Q4 (overhead > accept rate) — included as an honest negative result.

---

## Measurement protocol (summary)

- Same 3 prompts, `max_tokens=128`, `temperature=0`
- 1 warmup + 3 timed runs **per prompt** (warmup excluded from stats)
- Mean latency, mean tokens/sec, peak RSS (background sampler)
- 8 threads, `n_ctx=2048`

See `CASE_STUDY.md` for hardware dumps (`uname`, `nproc`, `free -h`) and full tables.

---

## License

Code in this repo: MIT. Model weights: Apache-2.0 (Qwen) via bartowski GGUF redistributions — download from Hugging Face; not redistributed here.
