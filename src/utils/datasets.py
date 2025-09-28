from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.utils import CLASSES
from src.utils.audio_arrays import per_example_zscore

@dataclass
class PathResolver:
    project_root: Path
    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.project_root / p)

class _BaseMelDataset(Dataset):
    def __init__(self, per_example_norm: bool = True):
        self.per_example_norm = per_example_norm

    def _load_mel(self, mel_path: Path) -> np.ndarray:
        x = np.load(mel_path, allow_pickle=False).astype(np.float32)  # (2, 128, T)
        if self.per_example_norm:
            x = per_example_zscore(x)
        return x

class SimpleMelNpyDataset(_BaseMelDataset):
    """Manifest CSV: filepath,label  (single-label)."""
    def __init__(self, manifest_csv: str, label_to_idx: Dict[str,int] | None = None,
                 per_example_norm: bool = True):
        super().__init__(per_example_norm)
        df = pd.read_csv(manifest_csv)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        self.df = df.reset_index(drop=True)

        if label_to_idx is None:
            labels = sorted(self.df["label"].unique())
            self.label_to_idx = {lbl:i for i,lbl in enumerate(labels)}
        else:
            self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {v:k for k,v in self.label_to_idx.items()}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = torch.from_numpy(self._load_mel(Path(row["filepath"])))
        y = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return x, y

class IRMASTestWindowDataset(_BaseMelDataset):
    """Multi-label windows; returns (mel, multi_hot, clip_id, path)."""
    def __init__(self, manifest_csv: Path, project_root: Path, class_names: List[str],
                 per_example_norm: bool = True):
        super().__init__(per_example_norm)
        self.resolver = PathResolver(project_root)
        self.class_names = list(class_names)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

        df = pd.read_csv(manifest_csv)
        required = {"filepath", "label_multi"}
        if not required.issubset(df.columns):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        df["filepath"] = df["filepath"].astype(str)
        df["label_multi"] = df["label_multi"].astype(str)
        df["clip_id"] = (df["irmas_filename"].astype(str)
                         if "irmas_filename" in df.columns
                         else df["filepath"].map(lambda p: Path(p).stem))
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        mel_path = self.resolver.resolve(row["filepath"])
        if not mel_path.exists():
            raise FileNotFoundError(mel_path)
        mel = torch.from_numpy(self._load_mel(mel_path))

        bits = row["label_multi"].strip()
        positives = [CLASSES[i] for i, ch in enumerate(bits[:len(CLASSES)]) if ch == "1"]
        target = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in positives:
            idx = self.label_to_idx.get(label)
            if idx is not None:
                target[idx] = 1.0
        return mel, target, row["clip_id"], str(mel_path)