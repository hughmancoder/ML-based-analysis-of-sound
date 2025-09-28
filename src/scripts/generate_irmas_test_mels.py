#!/usr/bin/env python3
"""
IRMAS TEST → slide 3.0 s windows, save 2-ch mel-dB .npy, emit manifest.

- Input:  --irmas_test_dir  (expects .wav with sibling .txt labels)
- Output CSV columns: filepath,label_multi,irmas_filename,start_ms

"""

from __future__ import annotations
import argparse, csv, sys, traceback, math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf

# Assumes utils.* is importable via PYTHONPATH (no hardcoded sys.path edits here).
from utils.utils import CLASSES
from utils.mel_utils import (
    _compute_starts,      # (clip_len_s, win_s, stride_s) -> start times (s)
    _hash_path,           # stable short hash for filenames
    _label_from_txt,      # reads sibling .txt & returns 11-bit multi-label as string
    _load_segment_stereo, # (path, sr, start_s, dur_s) -> (2, L) float32
    _stereo_to_mel,       # (stereo, sr, n_mels, win_ms, hop_ms, fmin, fmax) -> (2, n_mels, T)
    _safe_relpath,        # relpath to project root when possible
)

def main():
    ap = argparse.ArgumentParser(description="IRMAS TEST → mel windows + manifest")
    ap.add_argument("--irmas_test_dir", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--mel_manifest_out", required=True)

    # Defaults chosen to hit (2, 196, 301) if HOP_MS=10.0 and center=True in mel utils
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=196)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--stride_s", type=float, default=1.5)  # ~50% overlap

    # For relpath emission; fallback to absolute paths if not under this root.
    ap.add_argument("--project_root", type=str, default=".")

    args = ap.parse_args()

    test_root   = Path(args.irmas_test_dir)
    cache_root  = Path(args.cache_root).resolve()
    out_csv     = Path(args.mel_manifest_out)
    project_root = Path(args.project_root).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(test_root.rglob("*.wav"))
    if not wavs:
        print(f"[ERROR] No .wav under {test_root}", file=sys.stderr); sys.exit(2)


    rows: List[List[str]] = []
    n_ok = n_fail = n_windows = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(wavs, desc=f"Generating test mels")
    except Exception:
        iterator = wavs

    for wav_path in iterator:
        try:
            irmas_filename = wav_path.name
            label_multi = _label_from_txt(wav_path.with_suffix(".txt"))

            info = sf.info(str(wav_path))
            clip_len_s = float(info.frames) / float(info.samplerate)

            starts = _compute_starts(clip_len_s, win_s=args.dur, stride_s=args.stride_s)

            track_id = wav_path.stem
            wav_hash = _hash_path(str(wav_path.resolve()))

            for start_s in starts:
                stereo = _load_segment_stereo(wav_path, args.sr, start_s, args.dur)
                mel = _stereo_to_mel(
                    stereo, args.sr, args.n_mels, args.win_ms, args.hop_ms, args.fmin, args.fmax
                )  # (2, n_mels, T)

                # C, F, T = mel.shape
                # print("INFO mel shape", mel.shape)
                

                tag = (
                    f"sr{args.sr}_dur{args.dur}_m{args.n_mels}"
                    f"_w{int(args.win_ms)}_h{int(args.hop_ms)}_s{int(round(start_s*1000))}"
                )
                fn  = f"{track_id}__{wav_hash}__{tag}.npy"
                out_path = (cache_root / fn).resolve()
                np.save(out_path, mel.astype(np.float32))

                rel = _safe_relpath(out_path, project_root)
                rows.append([rel, label_multi, irmas_filename, int(round(start_s * 1000))])
                n_windows += 1

            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed {wav_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Write manifest (protect label_multi width)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label_multi", "irmas_filename", "start_ms"])
        w.writerows(rows)

    df = pd.read_csv(out_csv, dtype={"label_multi": "string"}, keep_default_na=False)
    bad = df.index[df["label_multi"].str.len() != len(CLASSES)].tolist()
    if bad:
        bad_csv = out_csv.with_suffix(".bad_rows.csv")
        df.iloc[bad].to_csv(bad_csv, index=False)
        print(f"[WARN] Bad label width rows written to {bad_csv}", file=sys.stderr)

    print(f"\nCache root: {cache_root}")
    print(f"Manifest:   {out_csv}")
    print(f"Clips OK: {n_ok} | Clips failed: {n_fail} | Windows: {n_windows}")
    sys.exit(0)

if __name__ == "__main__":
    main()