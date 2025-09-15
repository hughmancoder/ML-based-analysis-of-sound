from __future__ import annotations
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path

from src.audio.features import (
    load_audio_stereo, ensure_duration, calc_fft_hop, mel_stereo3_from_stereo
)

class MelDataset(Dataset):
    """
    Expects a manifest CSV with columns: filepath,label
    - If cache_root is set, will try to load/save (3,H,W) mel .npy files there.
    """
    def __init__(self, manifest_csv, label_to_idx: dict,
                 cache_root: str|None = None,
                 sr=44100, duration_s=3.0, n_mels=128, win_ms=30.0, hop_ms=10.0,
                 fmin=30.0, fmax=None):
        self.df = pd.read_csv(manifest_csv)
        self.label_to_idx = label_to_idx
        self.cache_root = Path(cache_root) if cache_root else None

        self.sr = sr
        self.duration_s = duration_s
        self.n_mels = n_mels
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr/2
        self.n_fft, self.hop = calc_fft_hop(sr, win_ms, hop_ms)

    def __len__(self): return len(self.df)

    def _cache_path(self, wav_path: Path):
        if not self.cache_root: return None
        rel = wav_path.with_suffix(".npy").name
        return self.cache_root / rel

    def _compute_mel(self, wav_path: Path):
        stereo = load_audio_stereo(wav_path, self.sr)              # (2,T)
        stereo = ensure_duration(stereo, self.sr, self.duration_s) # exact (2, T*)
        mel = mel_stereo3_from_stereo(stereo, self.sr, self.n_fft, self.hop,
                                      self.n_mels, self.fmin, self.fmax)     # (3,H,W)
        return mel.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = Path(row['filepath'])
        y_str = row['label'].strip().lower()
        y = self.label_to_idx[y_str]

        cpath = self._cache_path(wav)
        if cpath and cpath.exists():
            mel = np.load(cpath)
        else:
            mel = self._compute_mel(wav)
            if cpath:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(cpath, mel)

        x = torch.from_numpy(mel)  # (3,H,W)
        y = torch.tensor(y, dtype=torch.long)
        return x, y