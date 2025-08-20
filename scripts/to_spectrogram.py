#!/usr/bin/env python3
import argparse, pathlib
import numpy as np
from PIL import Image
import librosa

def mel_db(y, sr, n_fft, win_ms, hop_ms, window, n_mels, top_db):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        win_length=int(sr*win_ms/1000),
        hop_length=int(sr*hop_ms/1000),
        window=window, n_mels=n_mels, power=2.0, center=True,
    )
    return librosa.power_to_db(S, ref=np.max, top_db=top_db)

def stft_db(y, sr, n_fft, win_ms, hop_ms, window, top_db):
    D = librosa.stft(
        y=y, n_fft=n_fft,
        win_length=int(sr*win_ms/1000),
        hop_length=int(sr*hop_ms/1000),
        window=window, center=True,
    )
    S = np.abs(D)**2
    return librosa.power_to_db(S, ref=np.max, top_db=top_db)

def to_png(chw, out_path, size):
    out_path.parent.mkdir(parents=True, exist_ok=True)   # ensure dir exists
    vmin, vmax = float(chw.min()), float(chw.max())
    x = (255.0*(chw - vmin)/(vmax - vmin + 1e-8)).astype(np.uint8)
    img = np.transpose(x, (1, 2, 0))  # H, W, C
    Image.fromarray(img).resize((size, size), resample=Image.BICUBIC).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Input audio (.wav/.mp3/.flac)")
    ap.add_argument("out_png", help="Output PNG")
    ap.add_argument("--spec", choices=["mel","stft"], default="mel")
    ap.add_argument("--mode", choices=["mono","stereo3"], default="stereo3")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--window", default="boxcar")  # or "hann"
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--top_db", type=float, default=80.0)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out_png)

    # Robust loader: handles MP3/WAV/FLAC. mono=False keeps channels.
    y, sr = librosa.load(args.audio, sr=args.sr, mono=False)
    # librosa returns shape [n] for mono, [channels, n] for multi
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)  # duplicate for "stereo"
    else:
        # ensure 2 channels by duplicating the first if needed
        if y.shape[0] == 1:
            y = np.vstack([y, y])

    if args.spec == "stft":
        y_mono = y.mean(axis=0)
        S_db = stft_db(y_mono, sr, args.n_fft, args.win_ms, args.hop_ms, args.window, args.top_db)
        chw = np.expand_dims(S_db, 0)  # [1, F, T]
    else:
        if args.mode == "mono":
            y_mono = y.mean(axis=0)
            M_db = mel_db(y_mono, sr, args.n_fft, args.win_ms, args.hop_ms, args.window, args.n_mels, args.top_db)
            chw = np.expand_dims(M_db, 0)
        else:
            L_db = mel_db(y[0], sr, args.n_fft, args.win_ms, args.hop_ms, args.window, args.n_mels, args.top_db)
            R_db = mel_db(y[1], sr, args.n_fft, args.win_ms, args.hop_ms, args.window, args.n_mels, args.top_db)
            mean_db = 0.5*(L_db + R_db)
            Delta = librosa.feature.delta(mean_db)
            chw = np.stack([L_db, R_db, Delta], axis=0)  # [3, mel, T]

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)  # tile to 3 channels for VGG
    to_png(chw, out_path, args.img_size)
    print(f"Saved {args.spec.upper()} spectrogram → {out_path}")

if __name__ == "__main__":
    main()