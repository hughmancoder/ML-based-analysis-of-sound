from __future__ import annotations
from typing import Tuple
import hashlib
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf, librosa

def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:10]

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

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
    if stereo.shape[0] == 1:
        stereo = np.vstack([stereo, stereo])
    elif stereo.shape[0] > 2:
        stereo = stereo[:2]
    return stereo

def ensure_duration(stereo: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    C, T = stereo.shape; target = int(round(sr * duration_s))
    if T >= target: return stereo[:, :target]
    return np.pad(stereo, ((0,0),(0, target-T)))

def calc_fft_hop(sr: int, win_ms: float, hop_ms: float) -> Tuple[int,int,int]:
    win_length = int(round(sr * (win_ms/1000.0)))
    hop = int(round(sr * (hop_ms/1000.0)))
    n_fft = _next_pow2(win_length)
    return n_fft, hop, win_length

def mel_stereo2_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int, win_length: int,
                            n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or (sr/2)
    feats = []
    for ch in (0, 1):
        S = librosa.feature.melspectrogram(y=stereo[ch], sr=sr, n_fft=n_fft,
                                           hop_length=hop, win_length=win_length, window="hann",
                                           n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=True)
        S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        feats.append(S_db)
    return np.stack(feats, axis=0)  # (2, n_mels, T)

class MelDataset(Dataset):
    """
    Reads RAW-audio manifest (filepath,label). If cache_root is set, saves/loads
    (2, n_mels, T) .npy files in the SAME pattern as precache_mels.py:
      <cache_root>/<label>/<stem>__<hash>__sr{sr}_dur{dur}_m{M}_w{W}_h{H}.npy
    """
    def __init__(self, manifest_csv: str, label_to_idx: dict,
                 cache_root: str | None = None,
                 sr=44100, duration_s=3.0, n_mels=128, win_ms=30.0, hop_ms=10.0,
                 fmin=20.0, fmax=None):
        self.df = pd.read_csv(manifest_csv)
        self.label_to_idx = label_to_idx
        self.cache_root = Path(cache_root) if cache_root else None

        self.sr = sr; self.duration_s = duration_s
        self.n_mels = n_mels; self.win_ms = win_ms; self.hop_ms = hop_ms
        self.fmin = fmin; self.fmax = fmax if fmax is not None else sr/2
        self.n_fft, self.hop, self.win_length = calc_fft_hop(sr, win_ms, hop_ms)

    def __len__(self): return len(self.df)

    def _cache_path(self, wav_path: Path, label: str):
        if not self.cache_root: return None
        stem = wav_path.stem
        tag  = f"sr{self.sr}_dur{self.duration_s}_m{self.n_mels}_w{int(self.win_ms)}_h{int(self.hop_ms)}"
        fn   = f"{stem}__{_hash_path(str(wav_path))}__{tag}.npy"
        return self.cache_root / label / fn

    def _compute_mel(self, wav_path: Path):
        stereo = load_audio_stereo(wav_path, self.sr)
        stereo = ensure_duration(stereo, self.sr, self.duration_s)
        mel = mel_stereo2_from_stereo(stereo, self.sr,
                                      n_fft=self.n_fft, hop=self.hop, win_length=self.win_length,
                                      n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return mel.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = Path(row["filepath"])
        label = row["label"].strip().lower()
        y = self.label_to_idx[label]

        cpath = self._cache_path(wav, label)
        if cpath and cpath.exists():
            mel = np.load(cpath)
        else:
            mel = self._compute_mel(wav)
            if cpath:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(cpath, mel)
        return torch.from_numpy(mel), torch.tensor(y, dtype=torch.long)