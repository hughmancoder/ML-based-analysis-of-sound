# src/audio/features.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Iterable
import numpy as np
import soundfile as sf
import librosa

# ---------- IO helpers ----------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def is_audio(path: Path, exts: Iterable[str] = AUDIO_EXTS) -> bool:
    return path.suffix.lower() in exts

def load_audio_stereo(path: Path, target_sr: int) -> np.ndarray:
    """
    Returns (C, T) float32. Will have C=1 or C=2.
    Uses soundfile for decode, then librosa.resample per-channel.
    """
    x, sr_in = sf.read(str(path), always_2d=True)  # (T, C)
    x = x.astype(np.float32, copy=False)
    chans = []
    for c in range(x.shape[1]):
        y = librosa.resample(x[:, c], orig_sr=sr_in, target_sr=target_sr) if sr_in != target_sr else x[:, c]
        chans.append(y)
    T = max(len(ch) for ch in chans)
    chans = [np.pad(ch, (0, T - len(ch))) for ch in chans]
    return np.stack(chans, axis=0)  # (C, T)

# ---------- Length / framing ----------
def ensure_duration(samples: np.ndarray, sr: int, duration_s: float, val_mode: bool = False) -> np.ndarray:
    """
    Enforce exact duration in samples by crop or pad (zeros) on (C,T) or (T,C) or (T,).
    Returns (C,T).
    """
    # reshape to (C,T)
    if samples.ndim == 1:
        samples = samples[None, :]
    elif samples.shape[0] < samples.shape[1]:  # maybe (T,C)
        samples = samples.T  # -> (C,T)

    C, T = samples.shape
    target_len = int(round(sr * duration_s))
    if T >= target_len:
        start = 0 if val_mode else np.random.randint(0, T - target_len + 1)
        out = samples[:, start:start + target_len]
    else:
        pad = target_len - T
        out = np.pad(samples, ((0, 0), (0, pad)))
    return out

def calc_fft_hop(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int]:
    n_fft = int(round(sr * (win_ms / 1000.0)))
    hop = int(round(sr * (hop_ms / 1000.0)))
    return n_fft, hop

def expected_frames(duration_s: float, hop_ms: float) -> int:
    return int(np.ceil(duration_s * 1000.0 / hop_ms))

def crop_or_pad_time(mel: np.ndarray, target_W: int) -> np.ndarray:
    """
    mel: (C,H,W)
    Returns (C,H,target_W) via random crop (train) or center crop (if W > target)
    and edge-pad if W < target. Randomness handled outside if needed.
    """
    C, H, W = mel.shape
    if W == target_W:
        return mel
    if W > target_W:
        start = np.random.randint(0, W - target_W + 1)
        return mel[:, :, start:start + target_W]
    # W < target_W
    padW = target_W - W
    return np.pad(mel, ((0, 0), (0, 0), (0, padW)), mode="edge")

# ---------- Mel spectrograms ----------
def _mel_db_mono(wave: np.ndarray, sr: int, n_fft: int, hop: int,
                 n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    return librosa.power_to_db(S, ref=np.max)  # (H,W)

def mel_mono_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int,
                         n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    stereo: (C,T), C in {1,2}
    returns (1,H,W)
    """
    mono = stereo.mean(axis=0)
    mel = _mel_db_mono(mono, sr, n_fft, hop, n_mels, fmin, fmax)
    return mel[None, :, :]  # (1,H,W)

def mel_stereo3_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int,
                            n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    stereo: (C,T), C in {1,2}
    returns (3,H,W): [mel(L), mel(R), mel(M)]
    """
    if stereo.shape[0] == 1:
        L = R = stereo[0]
    else:
        L, R = stereo[0], stereo[1]
    M = (L + R) / 2.0
    mL = _mel_db_mono(L, sr, n_fft, hop, n_mels, fmin, fmax)
    mR = _mel_db_mono(R, sr, n_fft, hop, n_mels, fmin, fmax)
    mM = _mel_db_mono(M, sr, n_fft, hop, n_mels, fmin, fmax)
    return np.stack([mL, mR, mM], axis=0)  # (3,H,W)

# ---------- Normalization ----------
def standardize(mel: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = mel.mean()
    s = mel.std() + eps
    return (mel - m) / s

# ---------- Simple image helper (optional) ----------
def norm01(a: np.ndarray) -> np.ndarray:
    mn, mx = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float32) if mx - mn < 1e-6 else (a - mn) / (mx - mn)
