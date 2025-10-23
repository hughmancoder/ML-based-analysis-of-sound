#!/usr/bin/env python3
"""
Generate mel-spectrogram windows and emit a manifest CSV.

Each row in the manifest contains:
  filepath, labels, filename, start_ms, dataset

- `labels` is a semicolon-separated list of class names (e.g. "sax;pia")
  → no one-hot encoding is used.
- Audio windows are saved as 2-channel mel-spectrogram .npy files.
"""

from __future__ import annotations
import argparse, csv, sys, traceback
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import soundfile as sf


from src.utils.utils import _label_from_txt
from utils.mel_utils import (
    _compute_starts,
    _hash_path,
    _load_segment_stereo,
    _stereo_to_mel,
    _safe_relpath,
)


def main():
    ap = argparse.ArgumentParser(description="Generate mel windows + manifest (label names only)")
    ap.add_argument("--input_dir", required=True, help="Root directory containing .wav files")
    ap.add_argument("--cache_root", required=True, help="Where to save generated mel .npy files")
    ap.add_argument("--manifest_out", required=True, help="Output CSV manifest path")
    ap.add_argument("--project_root", type=str, default=".", help="Project root for relative paths")
    ap.add_argument("--dataset_name", type=str, default="IRMAS")

    # Audio config
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=196)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--stride_s", type=float, default=1.5)
    args = ap.parse_args()

    input_root   = Path(args.input_dir)
    cache_root   = Path(args.cache_root).resolve()
    manifest_out = Path(args.manifest_out)
    project_root = Path(args.project_root).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(input_root.rglob("*.wav"))
    if not wavs:
        print(f"[ERROR] No .wav files under {input_root}", file=sys.stderr)
        sys.exit(2)

    rows: List[List[str]] = []
    n_ok = n_fail = n_windows = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(wavs, desc=f"Generating {args.dataset_name} mel windows")
    except Exception:
        iterator = wavs

    for wav_path in iterator:
        try:
            labels = _label_from_txt(wav_path.with_suffix(".txt"))  # list[str] or str
            if isinstance(labels, str):
                labels = [labels]

            info = sf.info(str(wav_path))
            clip_len_s = info.frames / float(info.samplerate)
            starts = _compute_starts(clip_len_s, win_s=args.dur, stride_s=args.stride_s)
            wav_hash = _hash_path(str(wav_path.resolve()))

            for start_s in starts:
                stereo = _load_segment_stereo(wav_path, args.sr, start_s, args.dur)
                mel = _stereo_to_mel(
                    stereo, args.sr, args.n_mels, args.win_ms, args.hop_ms, args.fmin, args.fmax
                )

                tag = (
                    f"sr{args.sr}_dur{args.dur}_m{args.n_mels}"
                    f"_w{int(args.win_ms)}_h{int(args.hop_ms)}_s{int(round(start_s * 1000))}"
                )
                out_path = (cache_root / f"{wav_path.stem}__{wav_hash}__{tag}.npy").resolve()
                np.save(out_path, mel.astype(np.float32))

                rel_path = _safe_relpath(out_path, project_root)
                rows.append([
                    rel_path,
                    ";".join(labels),         # label names only
                    wav_path.name,
                    int(round(start_s * 1000)),
                    args.dataset_name,
                ])
                n_windows += 1

            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed {wav_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    with open(manifest_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "labels", "filename", "start_ms", "dataset"])
        writer.writerows(rows)

    print(f"\nManifest written to: {manifest_out}")
    print(f"Cache root: {cache_root}")
    print(f"Clips OK: {n_ok} | Failed: {n_fail} | Windows: {n_windows}")


if __name__ == "__main__":
    main()