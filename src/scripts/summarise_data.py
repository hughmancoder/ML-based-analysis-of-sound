#!/usr/bin/env python3
"""
Summarise dataset under a root (default: data/audio/chinese_instruments).

Prints:
  - total number of clips
  - per-label clip counts (label = top-level folder name)
  - estimated duration (clips × 3.0s)

Usage:
    PYTHONPATH=src python -m data.scripts.summarise_data [--root data/audio/chinese_instruments]
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

CLIP_SECONDS = 3.0

def human_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m or (h and s): parts.append(f"{m}m")
    if s and not h: parts.append(f"{s}s")
    return " ".join(parts) if parts else "0s"

def count_wavs(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not root.exists():
        print(f"[warn] root does not exist: {root}")
        return counts
    # labels are immediate subdirectories
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = 0
        for wav in label_dir.rglob("*.wav"):
            n += 1
        counts[label_dir.name] = n
    return counts

def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise Chinese instruments dataset.")
    ap.add_argument("--root", type=Path, default=Path("data/audio/chinese_instruments"),
                    help="Root dataset directory (default: data/audio/chinese_instruments)")
    args = ap.parse_args()

    counts = count_wavs(args.root)
    total = sum(counts.values())
    total_sec = total * CLIP_SECONDS

    print(f"Root: {args.root}")
    print(f"Total clips: {total}  (~{human_time(total_sec)})")
    if not counts:
        return
    print("\nPer-label clip counts:")
    width = max(len(k) for k in counts.keys())
    for label in sorted(counts.keys()):
        n = counts[label]
        dur = human_time(n * CLIP_SECONDS)
        print(f"  {label.ljust(width)}  {str(n).rjust(6)}  (~{dur})")

if __name__ == "__main__":
    main()
