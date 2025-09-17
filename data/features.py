from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np, soundfile as sf, librosa

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
    C, T = stereo.shape
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

def norm01(a: np.ndarray) -> np.ndarray:
    mn, mx = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float32) if mx - mn < 1e-6 else (a - mn) / (mx - mn)

def mel_stereo2_from_stereo(
    stereo: np.ndarray, sr: int, n_fft: int, hop: int,
    n_mels: int, fmin: float = 30.0, fmax: float | None = None
) -> np.ndarray:
    C, T = stereo.shape
    assert C >= 2, "expected stereo (2,T)"
    fmax = fmax if fmax is not None else sr / 2
    mels = []
    for ch in (0, 1):
        S = librosa.feature.melspectrogram(
            y=stereo[ch], sr=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
        )
        S_db = _power_to_db_safe(S)
        S01 = norm01(S_db).astype(np.float32)  # (H,W)
        mels.append(S01)
    return np.stack(mels, axis=0)  # (2,H,W)