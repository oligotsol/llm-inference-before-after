#!/usr/bin/env python3
"""Print a before/after comparison table from results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(label: str) -> dict:
    for p in RESULTS.glob("*.json"):
        if p.name == "comparison.json":
            continue
        d = json.loads(p.read_text())
        if d.get("label") == label:
            return d
    raise SystemExit(f"Missing results for label={label}")


def main():
    before = load("before_f16")
    after = load("after_q4_k_m")
    mid = None
    pld = None
    try:
        mid = load("mid_q8_0")
    except SystemExit:
        pass
    try:
        pld = load("after_q4_k_m_pld")
    except SystemExit:
        pass

    def row(name, d):
        return (
            f"| {name} | {d['model_file']} | {d['mean_latency_s']:.3f} | "
            f"{d['mean_tokens_per_sec']:.2f} | {d['mean_peak_rss_mb']:.1f} | "
            f"{d['model_file_size_mb']:.1f} |"
        )

    print("## Measured results\n")
    print("| Config | GGUF | Mean latency (s) | Mean tok/s | Peak RSS (MB) | File (MB) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    print(row("BEFORE F16", before))
    if mid:
        print(row("MID Q8_0", mid))
    print(row("AFTER Q4_K_M", after))
    if pld:
        print(row("AFTER Q4+PLD", pld))

    lat = (before["mean_latency_s"] - after["mean_latency_s"]) / before["mean_latency_s"] * 100
    mem = (before["mean_peak_rss_mb"] - after["mean_peak_rss_mb"]) / before["mean_peak_rss_mb"] * 100
    tps = (after["mean_tokens_per_sec"] - before["mean_tokens_per_sec"]) / before["mean_tokens_per_sec"] * 100
    print()
    print(f"- Latency reduction (F16 → Q4_K_M): **{lat:.1f}%**")
    print(f"- Peak RSS reduction: **{mem:.1f}%**")
    print(f"- Throughput increase: **{tps:.1f}%**")


if __name__ == "__main__":
    main()
