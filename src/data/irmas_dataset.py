# src/data/irmas_dataset.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import random_channel_swap, specaugment
from src.audio.features import (
    load_audio_stereo, ensure_duration, calc_fft_hop, expected_frames,
    mel_mono_from_stereo, mel_stereo3_from_stereo, crop_or_pad_time, standardize
)

IRMAS_TO_INDEX = {k: i for i, k in enumerate(["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"])}

class IRMASMelDataset(Dataset):
    """
    Returns:
      X: (C,H,W) float32 (C=1 mono or C=3 stereo3)
      y: int64 class index
    """
    def __init__(self,
                 manifest_csv: Path,
                 sr: int = 22050,
                 duration_s: float = 3.0,
                 n_mels: int = 128,
                 win_ms: float = 30.0,
                 hop_ms: float = 10.0,
                 fmin: float = 30.0,
                 fmax: Optional[float] = None,
                 stereo3: bool = True,
                 augment: bool = False,
                 val_mode: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.sr = sr
        self.duration_s = duration_s
        self.n_mels = n_mels
        self.n_fft, self.hop = calc_fft_hop(sr, win_ms, hop_ms)
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2.0
        self.stereo3 = stereo3
        self.augment = augment
        self.val_mode = val_mode
        self.target_W = expected_frames(duration_s, hop_ms)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = Path(row["filepath"])
        label_str = str(row["label"]).strip().lower()
        y_idx = IRMAS_TO_INDEX[label_str]

        # (C,T) at self.sr, exact duration
        stereo = load_audio_stereo(wav_path, self.sr)
        stereo = ensure_duration(stereo, self.sr, self.duration_s, val_mode=self.val_mode)

        # mel (C,H,W)
        if self.stereo3:
            mel = mel_stereo3_from_stereo(stereo, self.sr, self.n_fft, self.hop, self.n_mels, self.fmin, self.fmax)
        else:
            mel = mel_mono_from_stereo(stereo, self.sr, self.n_fft, self.hop, self.n_mels, self.fmin, self.fmax)

        # size to fixed W, then optional augment
        mel = crop_or_pad_time(mel, self.target_W)

        if self.augment:
            # channel swap is only meaningful for stereo (first two channels are L/R)
            # Keep M (3rd channel) stable; swap 0 and 1 with p=0.5
            if self.stereo3:
                mel = random_channel_swap(mel, p=0.5)
            mel = specaugment(mel, F=12, T=24, num_freq_masks=1, num_time_masks=1)

        mel = standardize(mel)

        X = torch.from_numpy(mel).float()        # (C,H,W)
        y = torch.tensor(y_idx, dtype=torch.long) # single-label
        return X, y
