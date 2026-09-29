#!/usr/bin/env python3
"""
Benchmark llama.cpp GGUF inference: latency, tokens/sec, peak RSS.

Warmup runs are excluded from statistics. Peak RSS is sampled during generation.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from llama_cpp import Llama

PROMPTS = [
    "Explain how quantization reduces memory usage in large language models in two short paragraphs.",
    "Write a Python function that computes Fibonacci numbers iteratively, with a brief docstring.",
    "List five practical tips for optimizing CPU inference throughput with llama.cpp.",
]

SYSTEM = "You are a concise technical assistant. Answer clearly and directly."


def hardware_info() -> dict:
    uname = platform.uname()
    mem = psutil.virtual_memory()
    return {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": round(mem.total / (1024 * 1024), 1),
        "ram_available_mb": round(mem.available / (1024 * 1024), 1),
        "uname_a": subprocess.check_output(["uname", "-a"], text=True).strip(),
        "nproc": int(subprocess.check_output(["nproc"], text=True).strip()),
        "free_h": subprocess.check_output(["free", "-h"], text=True).strip(),
        "lscpu_model": _cpu_model(),
    }


def _cpu_model() -> str:
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if "Model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


class RssSampler:
    """Sample peak RSS of current process in a background thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid())

    def start(self):
        self.peak_rss = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                break
            time.sleep(self.interval)

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass
        return self.peak_rss


def build_messages(user: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def run_once(llm: Llama, prompt: str, max_tokens: int) -> dict:
    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=build_messages(prompt),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    elapsed = time.perf_counter() - t0
    peak = sampler.stop()
    usage = out.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    text = out["choices"][0]["message"]["content"]
    tps = completion_tokens / elapsed if elapsed > 0 and completion_tokens > 0 else 0.0
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_sec": tps,
        "peak_rss_bytes": peak,
        "peak_rss_mb": round(peak / (1024 * 1024), 2),
        "response_chars": len(text),
        "response_preview": text[:200].replace("\n", " "),
    }


def load_model(
    path: str,
    n_threads: int,
    n_ctx: int,
    draft_model=None,
) -> Llama:
    kwargs = dict(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        verbose=False,
        logits_all=False,
    )
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
    return Llama(**kwargs)


def main():
    ap = argparse.ArgumentParser(description="GGUF before/after inference benchmark")
    ap.add_argument("--model", required=True, help="Path to target GGUF")
    ap.add_argument("--draft", default=None, help="Unused legacy flag (kept for CLI compat)")
    ap.add_argument("--prompt-lookup", action="store_true",
                    help="Enable prompt-lookup speculative decoding (LlamaPromptLookupDecoding)")
    ap.add_argument("--pld-ngram", type=int, default=3)
    ap.add_argument("--pld-pred", type=int, default=8)
    ap.add_argument("--label", required=True, help="Run label, e.g. before_f16 or after_q4")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3, help="Timed runs per prompt (after warmup)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per prompt (excluded)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 0))
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        sys.exit(f"Model not found: {model_path}")

    draft_model = None
    speculative_kind = None
    if args.prompt_lookup:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        draft_model = LlamaPromptLookupDecoding(
            max_ngram_size=args.pld_ngram,
            num_pred_tokens=args.pld_pred,
        )
        speculative_kind = f"prompt_lookup(ngram={args.pld_ngram},pred={args.pld_pred})"

    print(f"Loading {model_path} (threads={args.threads}, ctx={args.n_ctx}, speculative={speculative_kind})...", flush=True)
    llm = load_model(model_path, args.threads, args.n_ctx, draft_model)
    speculative = speculative_kind is not None
    draft_path = speculative_kind  # reuse field for JSON

    # Baseline RSS after load (before generation)
    baseline_rss = psutil.Process(os.getpid()).memory_info().rss

    all_runs = []
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n=== Prompt {pi + 1}/{len(PROMPTS)} ===", flush=True)
        for w in range(args.warmup):
            print(f"  warmup {w + 1}/{args.warmup}...", flush=True)
            run_once(llm, prompt, args.max_tokens)
        for r in range(args.runs):
            print(f"  timed run {r + 1}/{args.runs}...", flush=True)
            result = run_once(llm, prompt, args.max_tokens)
            result["prompt_index"] = pi
            result["run_index"] = r
            result["prompt_preview"] = prompt[:80]
            all_runs.append(result)
            print(
                f"    {result['elapsed_s']:.3f}s  "
                f"{result['completion_tokens']} tok  "
                f"{result['tokens_per_sec']:.2f} tok/s  "
                f"peak RSS {result['peak_rss_mb']:.1f} MB",
                flush=True,
            )

    latencies = [x["elapsed_s"] for x in all_runs]
    tps = [x["tokens_per_sec"] for x in all_runs]
    peaks = [x["peak_rss_mb"] for x in all_runs]
    completion_tokens = [x["completion_tokens"] for x in all_runs]

    summary = {
        "label": args.label,
        "model_path": model_path,
        "model_file": os.path.basename(model_path),
        "model_file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "draft_path": draft_path,
        "speculative": speculative,
        "max_tokens": args.max_tokens,
        "n_runs_per_prompt": args.runs,
        "n_warmup_per_prompt": args.warmup,
        "n_prompts": len(PROMPTS),
        "n_timed_runs_total": len(all_runs),
        "n_threads": args.threads,
        "n_ctx": args.n_ctx,
        "temperature": 0.0,
        "baseline_rss_after_load_mb": round(baseline_rss / (1024 * 1024), 2),
        "mean_latency_s": statistics.mean(latencies),
        "stdev_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_tokens_per_sec": statistics.mean(tps),
        "stdev_tokens_per_sec": statistics.stdev(tps) if len(tps) > 1 else 0.0,
        "mean_peak_rss_mb": statistics.mean(peaks),
        "max_peak_rss_mb": max(peaks),
        "mean_completion_tokens": statistics.mean(completion_tokens),
        "prompts": PROMPTS,
        "runs": all_runs,
        "hardware": hardware_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().astimezone().isoformat(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print(
        f"SUMMARY [{args.label}] mean latency={summary['mean_latency_s']:.3f}s  "
        f"mean tok/s={summary['mean_tokens_per_sec']:.2f}  "
        f"mean peak RSS={summary['mean_peak_rss_mb']:.1f} MB  "
        f"max peak RSS={summary['max_peak_rss_mb']:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
