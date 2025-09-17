#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


import pandas as pd
from torch.utils.data import DataLoader
from data.mel_dataset import MelDataset

def build_label_maps(csv_path: Path):
    df = pd.read_csv(csv_path)
    classes = sorted(df["label"].astype(str).str.lower().unique())
    label_to_idx = {c:i for i,c in enumerate(classes)}
    return label_to_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    label_to_idx = build_label_maps(Path(args.manifest_csv))
    ds = MelDataset(
        manifest_csv=args.manifest_csv,
        label_to_idx=label_to_idx,
        cache_root=str(cache_root),
        sr=args.sr, duration_s=args.dur,
        n_mels=args.n_mels, win_ms=args.win_ms, hop_ms=args.hop_ms
    )

    # Trigger computation by iterating; DataLoader workers will fan out CPU work.
    # We don't need the outputs; just reading items writes cache.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=False)

    for _ in loader:
        pass

    # Save label map for training time
    with open(cache_root / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"Cached mels to {cache_root}  (count={len(ds)})")

if __name__ == "__main__":
    main()