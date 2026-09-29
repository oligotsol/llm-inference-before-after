#!/usr/bin/env python3
"""Download bartowski Qwen2.5-0.5B-Instruct GGUF variants used in this case study."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
FILES = {
    "f16": "Qwen2.5-0.5B-Instruct-f16.gguf",
    "q8": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    "q4": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="/workspace/models",
        help="Directory to store GGUF files (gitignored)",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=list(FILES) + ["all"],
        default=["all"],
    )
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = list(FILES) if "all" in args.variants else args.variants
    for v in variants:
        name = FILES[v]
        print(f"Downloading {name} ...")
        path = hf_hub_download(
            repo_id=REPO,
            filename=name,
            local_dir=str(out),
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
