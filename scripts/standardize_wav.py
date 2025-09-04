#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Iterable
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

AUDIO_EXTS: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

def is_audio(p: Path, exts: Iterable[str] = AUDIO_EXTS) -> bool:
    return p.suffix.lower() in exts

def load_and_standardize(in_path: Path, sr_out: int, duration_s: float) -> np.ndarray:
    """
    Returns (C=2, T) float32 at sr_out, exactly duration_s long (trim/pad).
    - Decodes with soundfile
    - Resamples per channel with librosa
    - Ensures stereo (duplicate mono to L/R)
    """
    x, sr_in = sf.read(str(in_path), always_2d=True)  # (T, C_in)
    x = x.astype(np.float32, copy=False)

    # resample per channel if needed
    chans = []
    for c in range(x.shape[1]):
        y = librosa.resample(x[:, c], orig_sr=sr_in, target_sr=sr_out) if sr_in != sr_out else x[:, c]
        chans.append(y)
    T = max(len(ch) for ch in chans)
    chans = [np.pad(ch, (0, T - len(ch))) for ch in chans]
    stereo = np.stack(chans, axis=0)  # (C_in, T)

    # ensure stereo
    if stereo.shape[0] == 1:
        stereo = np.vstack([stereo, stereo])  # duplicate mono -> L/R

    # exact duration
    target_len = int(round(sr_out * duration_s))
    if stereo.shape[1] >= target_len:
        start = 0  # deterministic crop; change to random if you prefer
        stereo = stereo[:, start:start + target_len]
    else:
        pad = target_len - stereo.shape[1]
        stereo = np.pad(stereo, ((0, 0), (0, pad)))

    return stereo.astype(np.float32, copy=False)  # (2, T)

def write_pcm16_wav(out_path: Path, stereo: np.ndarray, sr: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # clamp to [-1, 1] then write as 16-bit PCM
    y = np.clip(stereo.T, -1.0, 1.0)  # soundfile expects (T, C)
    sf.write(str(out_path), y, sr, subtype="PCM_16")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="File or directory containing audio")
    ap.add_argument("output", type=Path, help="Output directory root for standardized WAVs")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--duration_s", type=float, default=3.0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src = args.input.resolve()
    dst_root = args.output.resolve()
    if src.is_file():
        rel = src.name
        out = dst_root / Path(rel).with_suffix(".wav")
        stereo = load_and_standardize(src, args.sr, args.duration_s)
        write_pcm16_wav(out, stereo, args.sr)
        if args.verbose: print(f"[prep] {src} -> {out}")
        return

    if not src.is_dir():
        print("Input path does not exist.", file=sys.stderr); sys.exit(1)

    files = [p for p in (src.rglob("*") if args.recursive else src.iterdir())
             if p.is_file() and is_audio(p)]
    if not files:
        print("No audio files found.", file=sys.stderr); sys.exit(2)

    for f in tqdm(files, desc="Standardizing"):
        rel_parent = f.parent.relative_to(src)
        out_dir = dst_root / rel_parent
        out_wav = out_dir / (f.stem + ".wav")
        stereo = load_and_standardize(f, args.sr, args.duration_s)
        write_pcm16_wav(out_wav, stereo, args.sr)
        if args.verbose: print(f"[prep] {f} -> {out_wav}")

if __name__ == "__main__":
    main()
