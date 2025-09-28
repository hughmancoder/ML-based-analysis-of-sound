#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

from utils.utils import CLASSES

def read_label_txt(txt_path: Path) -> str:
    present = set()
    if not txt_path.exists():
        # No annotation file → treat as zero vector (should not happen in IRMAS)
        return "0" * len(CLASSES)
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        k = line.strip().lower()
        if k in CLASSES:
            present.add(k)
    bits = ["1" if c in present else "0" for c in CLASSES]
    return "".join(bits)

def main():
    ap = argparse.ArgumentParser(description="Scan IRMAS test dir and emit CSV: filepath,label_multi")
    ap.add_argument("--irmas_test_dir", required=True, help="e.g., data/audio/IRMAS/IRMAS-TestingData-Part1")
    ap.add_argument("--out_csv", required=True, help="e.g., data/manifests/irmas_test.csv")
    args = ap.parse_args()

    root = Path(args.irmas_test_dir)
    rows = []
    for wav in root.rglob("*.wav"):
        txt = wav.with_suffix(".txt")
        label_multi = read_label_txt(txt)
        rows.append({"filepath": str(wav), "label_multi": label_multi})

    df = pd.DataFrame(rows).sort_values("filepath")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows → {args.out_csv}")

if __name__ == "__main__":
    main()
