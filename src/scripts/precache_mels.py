#!/usr/bin/env python3
"""_summary_
converts a WAV manifest to a mel spectrogram manifest by generating .npy array files
"""
from __future__ import annotations
import argparse, csv, sys, traceback
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Import the EXACT helpers you already use
from data.utils.mel_utils import (
    _hash_path,
    load_audio_stereo,
    ensure_duration,
    calc_fft_hop,
    mel_stereo2_from_stereo,
    _safe_relpath,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


def precache_one(wav_path: Path, label: str, cache_root: Path,
                 sr: int, dur: float, n_mels: int, win_ms: float, hop_ms: float,
                 fmin: float, fmax: Optional[float]) -> Path:
    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    stereo = load_audio_stereo(wav_path, target_sr=sr)
    stereo = ensure_duration(stereo, sr, dur)
    mel = mel_stereo2_from_stereo(
        stereo, sr,
        n_fft=n_fft, hop=hop, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )  # (2, n_mels, T)

    stem = wav_path.stem
    tag  = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}"
    fn   = f"{stem}__{_hash_path(str(wav_path))}__{tag}.npy"
    out_path = cache_root / label / fn
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel.astype(np.float32))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Precompute 2ch mel-dB tensors (.npy) from a WAV manifest and emit a new manifest.")
    ap.add_argument("--manifest_csv", required=True, type=str, help="CSV: filepath,label (paths to RAW WAVs).")
    ap.add_argument("--cache_root", required=True, type=str, help="Root folder for cached .npy.")
    ap.add_argument("--mel_manifest_out", required=True, type=str, help="Output CSV listing .npy paths and labels.")

    # Kept for CLI compatibility (not used directly here)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)

    # Mel params (must match your training pipeline)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)

    args = ap.parse_args()

    manifest_csv = Path(args.manifest_csv)
    cache_root   = Path(args.cache_root)
    out_csv      = Path(args.mel_manifest_out)

    if not manifest_csv.exists():
        print(f"ERROR: manifest_csv not found: {manifest_csv}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(manifest_csv)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if not {"filepath","label"}.issubset(df.columns):
        print("ERROR: manifest must have columns: filepath,label", file=sys.stderr)
        sys.exit(2)

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    rows_out: List[List[str]] = []
    n_ok, n_fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-caching mels"):
        wav_path = Path(row["filepath"])
        label = row["label"]
        try:
            if not wav_path.exists():
                raise FileNotFoundError(str(wav_path))
            npy_path = precache_one(
                wav_path, label, cache_root,
                sr=args.sr, dur=args.dur, n_mels=args.n_mels,
                win_ms=args.win_ms, hop_ms=args.hop_ms,
                fmin=args.fmin, fmax=args.fmax
            )
            rows_out.append([_safe_relpath(npy_path, cache_root), label])
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed: {wav_path} ({e})", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label"])
        w.writerows(rows_out)

    print(f"\nCache root: {cache_root}")
    print(f"Manifest written: {out_csv}")
    print(f"Success: {n_ok} | Failed: {n_fail}")

if __name__ == "__main__":
    main()
