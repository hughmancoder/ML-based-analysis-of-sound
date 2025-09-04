#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root so 'src' is importable

from src.audio.features import (
    is_audio, load_audio_stereo, ensure_duration, calc_fft_hop, expected_frames,
    mel_mono_from_stereo, mel_stereo3_from_stereo, norm01
)

# ---------- image save ----------
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def save_png(arr: np.ndarray, out_path: Path, img_size: int | None):
    ensure_parent(out_path)
    if arr.ndim == 2:       # (H,W)
        img = (255 * norm01(arr)).astype(np.uint8)
        pil = Image.fromarray(img, mode="L")
    elif arr.ndim == 3:     # (C,H,W) => choose (H,W) or map channels
        if arr.shape[0] == 1:          # mono mel
            img = (255 * norm01(arr[0])).astype(np.uint8)
            pil = Image.fromarray(img, mode="L")
        elif arr.shape[0] == 3:        # stereo3
            # convert (3,H,W) -> (H,W,3)
            rgb = np.transpose(arr, (1, 2, 0))
            img = (255 * norm01(rgb)).astype(np.uint8)
            pil = Image.fromarray(img, mode="RGB")
        else:
            raise ValueError("Expected channels 1 or 3 for image export")
    else:
        raise ValueError("Expected 2D or 3D array")
    if img_size is not None:
        pil = pil.resize((img_size, img_size), Image.BICUBIC)
    pil.save(out_path)

# ---------- core ----------
def mel_from_file(in_file: Path, out_png: Path, args):
    stereo = load_audio_stereo(in_file, args.sr)                  # (C,T)
    stereo = ensure_duration(stereo, args.sr, args.duration_s)    # exact length
    n_fft, hop = calc_fft_hop(args.sr, args.win_ms, args.hop_ms)
    fmax = args.fmax if args.fmax is not None else args.sr / 2.0

    if args.mode == "mono":
        mel = mel_mono_from_stereo(stereo, args.sr, n_fft, hop, args.n_mels, args.fmin, fmax)  # (1,H,W)
    elif args.mode == "stereo3":
        mel = mel_stereo3_from_stereo(stereo, args.sr, n_fft, hop, args.n_mels, args.fmin, fmax)  # (3,H,W)
    else:
        raise ValueError("mode must be 'mono' or 'stereo3'")

    save_png(mel, out_png, args.img_size)
    if args.save_npy:
        np.save(out_png.with_suffix(".npy"), mel.astype(np.float32))

    if args.verbose:
        print(f"[mel] {in_file} -> {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Audio file OR directory")
    ap.add_argument("output", type=Path,
                    help="If file: PNG path or directory; If dir: output root directory.")
    ap.add_argument("--mode", choices=["mono", "stereo3"], default="stereo3")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration_s", type=float, default=3.0)          # <— add duration to match dataset
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--fmin", type=float, default=30.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--save_npy", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()

    if inp.is_file():
        out_png = out if out.suffix.lower() == ".png" else (out / (inp.stem + ".png"))
        mel_from_file(inp, out_png, args)
        return

    if inp.is_dir():
        if out.suffix.lower() == ".png":
            print("When input is a directory, output must be a directory.", file=sys.stderr)
            sys.exit(1)

        files = [p for p in (inp.rglob("*") if args.recursive else inp.iterdir())
                 if p.is_file() and is_audio(p)]
        if not files:
            print("No audio files found.", file=sys.stderr)
            sys.exit(1)

        for f in files:
            rel_parent = f.parent.relative_to(inp)
            out_dir = out / rel_parent
            out_png = out_dir / (f.stem + ".png")
            mel_from_file(f, out_png, args)
        return

    print("Input path does not exist.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
