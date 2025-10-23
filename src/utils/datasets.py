from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.classes import IRMAS_CLASSES

def per_example_zscore(x: np.ndarray | torch.Tensor, eps: float = 1e-6):
    if isinstance(x, np.ndarray):
        mean = x.mean(axis=(1,2), keepdims=True)
        std  = x.std(axis=(1,2), keepdims=True).clip(min=eps)
        return (x - mean) / std
    # torch.Tensor path
    mean = x.mean(dim=(1,2), keepdim=True)
    std  = x.std(dim=(1,2), keepdim=True).clamp_min(eps)
    return (x - mean) / std

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

class SingleClassMelNpyDataset(_BaseMelDataset):
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
    """
    Multi-label windows; returns (mel, multi_hot, clip_id, path).

    Manifest (new schema) columns:
      - filepath : path to .npy mel (relative or absolute)
      - labels   : semicolon-separated names, e.g. "sax;pia" (can be empty "")
      - filename : original audio filename (optional)
      - start_ms : optional
      - dataset  : optional
    """
    def __init__(
        self,
        manifest_csv: Path,
        project_root: Path,
        class_names: List[str],
        per_example_norm: bool = True,
        unknown_label_policy: str = "warn",  # "ignore" | "warn" | "error"
    ):
        super().__init__(per_example_norm)
        self.resolver = PathResolver(project_root)
        self.class_names = list(class_names)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.unknown_label_policy = unknown_label_policy

        df = pd.read_csv(manifest_csv)
        required = {"filepath", "labels"}
        if not required.issubset(df.columns):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")

        # Normalize dtypes
        df["filepath"] = df["filepath"].astype(str)
        df["labels"] = df["labels"].fillna("").astype(str)

        # Parse "a;b;c" -> ["a","b","c"]
        def _parse_labels(s: str) -> list[str]:
            return [t.strip() for t in s.split(";") if t.strip()]

        df["labels_parsed"] = df["labels"].map(_parse_labels)

        # Derive a stable clip_id
        if "filename" in df.columns:
            df["clip_id"] = df["filename"].astype(str).map(lambda s: Path(s).stem)
        elif "irmas_filename" in df.columns:  # legacy fallback
            df["clip_id"] = df["irmas_filename"].astype(str).map(lambda s: Path(s).stem)
        else:
            def _infer_clip_id(p: str) -> str:
                stem = Path(p).stem
                # If your saved mels look like NAME__HASH__TAG.npy, keep NAME
                return stem.split("__", 1)[0]
            df["clip_id"] = df["filepath"].map(_infer_clip_id)

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_multi_hot(self, names: Sequence[str], row_idx: int) -> torch.Tensor:
        tgt = torch.zeros(len(self.class_names), dtype=torch.float32)
        unknowns = []
        for name in names:
            idx = self.label_to_idx.get(name)
            if idx is None:
                unknowns.append(name)
            else:
                tgt[idx] = 1.0

        if unknowns:
            if self.unknown_label_policy == "error":
                raise KeyError(
                    f"Unknown label(s) in row {row_idx}: {unknowns} "
                    f"(allowed: {self.class_names})"
                )
            elif self.unknown_label_policy == "warn":
                print(f"[WARN] Unknown label(s) {unknowns} in row {row_idx}; ignoring.",
                      file=sys.stderr)
        return tgt

    def __getitem__(self, index):
        row = self.df.iloc[index]
        mel_path = self.resolver.resolve(row["filepath"])
        if not mel_path.exists():
            raise FileNotFoundError(mel_path)

        mel = torch.from_numpy(self._load_mel(mel_path))  # (C, F, T)
        target = self._encode_multi_hot(row["labels_parsed"], index)
        return mel, target, row["clip_id"], str(mel_path)
