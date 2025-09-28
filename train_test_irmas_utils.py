
# TODO: migrate to utils
from __future__ import annotations
import argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys


def setup_seed(seed: int = 1337):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def pick_device() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

def pad_collate(batch):
    xs, ys, clip_ids, paths = zip(*batch)
    B = len(xs)
    C, F = xs[0].shape[:2]
    Tmax = max(x.shape[-1] for x in xs)
    padded = torch.zeros(B, C, F, Tmax, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        T = x.shape[-1]
        padded[i, :, :, :T] = x
    targets = torch.stack(ys).float()
    return padded, targets, list(clip_ids), list(paths)

# ------------------------
# Dataset & collate
# ------------------------
class SimpleMelNpyDataset(Dataset):
    """Manifest CSV: filepath,label. Each .npy: float32 (2, 128, T)."""
    def __init__(self, manifest_csv: str, label_to_idx: dict[str,int] | None = None,
                 train: bool = True, per_example_norm: bool = True):
        df = pd.read_csv(manifest_csv)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        self.df = df.reset_index(drop=True)

        if label_to_idx is None:
            labels = sorted(self.df["label"].unique())
            self.label_to_idx = {lbl:i for i,lbl in enumerate(labels)}
        else:
            self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {v:k for k,v in self.label_to_idx.items()}

        self.train = train
        self.per_example_norm = per_example_norm

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["filepath"], allow_pickle=False).astype(np.float32)  # (2,128,T)
        x = torch.from_numpy(x)
        if self.per_example_norm:
            mean = x.mean(dim=(1,2), keepdim=True)
            std  = x.std(dim=(1,2), keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
        y = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return x, y

# Note: not really necessary to pad here since mel tensors have same width
def pad_time_axes(batch, pad_value: float = 0.0):
    """
    Pads variable T to max T in batch.
    Returns X(B,2,128,Tmax), y(B,), lens(B,)
    """
    xs, ys = zip(*batch)
    B = len(xs); C, F = xs[0].shape[:2]
    Tmax = max(x.shape[2] for x in xs)
    # print("INFO: padding to Tmax =", Tmax)
    X = xs[0].new_full((B, C, F, Tmax), pad_value)
    lens = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(xs):
        T = x.shape[2]
        X[i, :, :, :T] = x
        lens[i] = T
    y = torch.stack(ys, 0)
    return X, y, lens


class IRMASTestWindowDataset(Dataset):
    def __init__(self, manifest_csv: Path, project_root: Path, class_names, per_example_norm: bool = True):
        self.project_root = project_root
        self.class_names = list(class_names)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.per_example_norm = per_example_norm
        df = pd.read_csv(manifest_csv)
        required = {'filepath', 'label_multi'}
        if not required.issubset(df.columns):
            raise ValueError(f'Manifest must contain columns: {sorted(required)}')
        df['filepath'] = df['filepath'].astype(str)
        df['label_multi'] = df['label_multi'].astype(str)
        if 'irmas_filename' in df.columns:
            df['clip_id'] = df['irmas_filename'].astype(str)
        else:
            df['clip_id'] = df['filepath'].map(lambda p: Path(p).stem)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / raw_path
        return path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        mel_path = self._resolve_path(row['filepath'])
        if not mel_path.exists():
            raise FileNotFoundError(mel_path)
        mel = np.load(mel_path, allow_pickle=False).astype(np.float32)
        if self.per_example_norm:
            mean = mel.mean(axis=(1, 2), keepdims=True)
            std = mel.std(axis=(1, 2), keepdims=True).clip(min=1e-6)
            mel = (mel - mean) / std
        bits = row['label_multi'].strip()
        positives = [CLASSES[i] for i, ch in enumerate(bits[:len(CLASSES)]) if ch == '1']
        target = np.zeros(len(self.class_names), dtype=np.float32)
        for label in positives:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0
        return torch.from_numpy(mel), torch.from_numpy(target), row['clip_id'], str(mel_path)



