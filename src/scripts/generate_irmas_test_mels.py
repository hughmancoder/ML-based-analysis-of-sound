#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys, traceback
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import soundfile as sf

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.utils import CLASSES
from utils.mel_utils import _hash_path, _load_segment_stereo, _stereo_to_mel, _safe_relpath

PROJECT_ROOT = SRC_ROOT.parent

def _label_from_txt(txt_path: Path) -> str:
    present = set()
    if txt_path.exists():
        for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            k = line.strip().lower()
            if k in CLASSES:
                present.add(k)
    return "".join("1" if c in present else "0" for c in CLASSES)

def _ensure_11_bits(label_multi: str, wav_path: Path) -> str:
    s = "".join(ch for ch in str(label_multi).strip() if ch in "01")
    return s if len(s) == len(CLASSES) else _label_from_txt(wav_path.with_suffix(".txt"))

def _compute_starts(clip_len_s: float, win_s: float, hop_s: float) -> List[float]:
    if hop_s <= 0:
        raise ValueError("--hop_s must be > 0 (got %.3f)" % hop_s)
    starts: List[float] = []
    s = 0.0
    eps = 1e-6
    # schedule forward strides where the full window fits
    while s + win_s <= clip_len_s + eps:
        starts.append(s)
        s += hop_s
    if not starts:
        starts = [0.0]
    # ensure tail coverage (if last window didn’t reach the end)
    tail_start = max(clip_len_s - win_s, 0.0)
    if starts[-1] + win_s < clip_len_s - eps and abs(tail_start - starts[-1]) > eps:
        starts.append(tail_start)
    # dedupe at millisecond precision
    starts = sorted(set(round(x, 3) for x in starts))
    return starts

def main():
    ap = argparse.ArgumentParser(
        description="IRMAS TEST → mel windows + minimal manifest: filepath,label_multi,irmas_filename"
    )
    ap.add_argument("--irmas_test_dir", required=True)
    ap.add_argument("--cache_root", required=True, help="Where .npy mels are written")
    ap.add_argument("--mel_manifest_out", required=True)

    # MUST MATCH TRAIN
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)

    # Window stride across long test clips: expressed in STFT frames (aligns to hop_ms grid)
    ap.add_argument(
        "--stride_frames",
        type=int,
        default=None,
        help="Stride between window starts, in mel/STFT frames. If not set, defaults to round( (dur/2) / (hop_ms/1000) )."
    )

    # (optional) write a tiny config for parity checks
    ap.add_argument("--write_config", action="store_true")

    args = ap.parse_args()

    # Derive hop_s (seconds) from hop_ms and stride_frames (if None, default to 50% overlap)
    frame_hop_s = args.hop_ms / 1000.0
    if args.stride_frames is None:
        # Default to ~50% overlap in time
        default_stride_frames = max(1, int(round((args.dur * 0.5) / frame_hop_s)))
        args.stride_frames = default_stride_frames
    window_stride_s = args.stride_frames * frame_hop_s

    test_root  = Path(args.irmas_test_dir)
    cache_root = Path(args.cache_root).resolve()
    out_csv    = Path(args.mel_manifest_out)
    cache_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Optional config sidecar to assert parity with train later
    if args.write_config:
        cfg = {
            "sr": args.sr, "dur": args.dur, "n_mels": args.n_mels,
            "win_ms": args.win_ms, "hop_ms": args.hop_ms,
            "fmin": args.fmin, "fmax": args.fmax,
            "window_stride_frames": args.stride_frames,
            "window_hop_s": window_stride_s,
            "classes": CLASSES,
        }
        (out_csv.with_suffix(".config.json")).write_text(json.dumps(cfg, indent=2))

    wavs = sorted(test_root.rglob("*.wav"))
    if not wavs:
        print(f"[ERROR] No .wav under {test_root}", file=sys.stderr); sys.exit(2)

    print(f"[INFO] Window stride: stride_frames={args.stride_frames}, hop_ms={args.hop_ms} -> window_stride_s={window_stride_s:.3f}s")

    rows: List[List[str]] = []
    bad_rows: List[dict] = []
    n_ok = n_fail = n_windows = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(wavs, desc="Generating test mel windows")
    except Exception:
        iterator = wavs

    for wav_path in iterator:
        try:
            label_multi = _ensure_11_bits(_label_from_txt(wav_path.with_suffix(".txt")), wav_path)
            if len(label_multi) != len(CLASSES):
                bad_rows.append({"reason":"label_len_mismatch_after_fix","wav_path":str(wav_path),"label_multi":label_multi})

            info = sf.info(str(wav_path))
            clip_len_s = float(info.frames) / float(info.samplerate)

            starts = _compute_starts(clip_len_s, win_s=args.dur, hop_s=window_stride_s)

            track_id = wav_path.stem
            wav_hash = _hash_path(str(wav_path.resolve()))

            for start_s in starts:
                stereo = _load_segment_stereo(wav_path, args.sr, start_s, args.dur)
                mel = _stereo_to_mel(stereo, args.sr, args.n_mels, args.win_ms, args.hop_ms, args.fmin, args.fmax)

                tag = f"sr{args.sr}_dur{args.dur}_m{args.n_mels}_w{int(args.win_ms)}_h{int(args.hop_ms)}_s{int(round(start_s*1000))}"
                fn  = f"{track_id}__{wav_hash}__{tag}.npy"
                out_path = (cache_root / fn).resolve()
                np.save(out_path, mel)

                mel_path = _safe_relpath(out_path, PROJECT_ROOT)

                rows.append([mel_path, label_multi, wav_path.name])
                n_windows += 1
            n_ok += 1

        except Exception as e:
            n_fail += 1
            bad_rows.append({"reason": f"exception:{type(e).__name__}", "wav_path": str(wav_path), "error": str(e)})
            print(f"[WARN] Failed: {wav_path} ({e})", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Write train-style minimal manifest
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label_multi","irmas_filename"])
        w.writerows(rows)

    # Validate label width using string dtype (preserve leading zeros)
    df = pd.read_csv(out_csv, dtype={"label_multi":"string"}, keep_default_na=False)
    bad = df.index[df["label_multi"].str.len() != len(CLASSES)].tolist()
    if bad:
        (out_csv.with_suffix(".bad_rows.csv")).write_text(
            df.iloc[bad].to_csv(index=False), encoding="utf-8"
        )
        print(f"[WARN] Some rows with bad label length logged to {out_csv.with_suffix('.bad_rows.csv')}", file=sys.stderr)

    print(f"\nCache root: {cache_root}")
    print(f"Manifest:   {out_csv}")
    print(f"Clips OK: {n_ok} | Clips failed: {n_fail} | Windows: {n_windows}")
    sys.exit(0)

if __name__ == "__main__":
    main()
