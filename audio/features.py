from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np, soundfile as sf, librosa

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

def is_audio(p: Path, exts: Iterable[str] = AUDIO_EXTS) -> bool:
    return p.suffix.lower() in exts

def load_audio_stereo(path: Path, target_sr: int) -> np.ndarray:
    x, sr_in = sf.read(str(path), always_2d=True)  # (T,C)
    x = x.astype(np.float32, copy=False)
    chans = []
    for c in range(x.shape[1]):
        y = librosa.resample(x[:, c], orig_sr=sr_in, target_sr=target_sr) if sr_in != target_sr else x[:, c]
        chans.append(y)
    T = max(len(ch) for ch in chans)
    chans = [np.pad(ch, (0, T-len(ch))) for ch in chans]
    stereo = np.stack(chans, axis=0)  # (C,T)
    if stereo.shape[0] == 1:  # ensure stereo
        stereo = np.vstack([stereo, stereo])
    return stereo

def ensure_duration(stereo: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    C,T = stereo.shape
    target = int(round(sr * duration_s))
    if T >= target:
        return stereo[:, :target]
    else:
        return np.pad(stereo, ((0,0),(0,target-T)))

def calc_fft_hop(sr: int, win_ms: float, hop_ms: float) -> Tuple[int,int]:
    n_fft = int(round(sr * (win_ms/1000.0)))
    hop   = int(round(sr * (hop_ms/1000.0)))
    return n_fft, hop

def _power_to_db_safe(P: np.ndarray) -> np.ndarray:
    P = np.maximum(P, 1e-10)
    ref = float(P.max()) if np.isfinite(P).all() else 1.0
    ref = 1.0 if ref <= 0 else ref
    return librosa.power_to_db(P / ref, top_db=80.0)

def mel_mono(wave: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    P = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    return _power_to_db_safe(P)

def mel_stereo3_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int,
                            n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    L, R = stereo[0], stereo[1]
    M = 0.5*(L+R)
    mL = mel_mono(L, sr, n_fft, hop, n_mels, fmin, fmax)
    mR = mel_mono(R, sr, n_fft, hop, n_mels, fmin, fmax)
    mM = mel_mono(M, sr, n_fft, hop, n_mels, fmin, fmax)
    # (3,H,W)
    return np.stack([mL, mR, mM], axis=0)

def norm01(a: np.ndarray) -> np.ndarray:
    mn, mx = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float32) if mx - mn < 1e-6 else (a - mn) / (mx - mn)