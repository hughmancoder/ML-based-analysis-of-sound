# src/utils/audio_arrays.py
from __future__ import annotations
import numpy as np
import torch

# ------------------------------------------------------
# Collate function for DataLoader when using windowed IRMAS datasets
# Pads variable-length time axis (last dimension) across a batch of spectrograms.
# Returns:
#   padded: (B, C, F, Tmax) tensor with zeros for padding
#   targets: stacked targets (shape depends on dataset: multi-hot or class index)
#   clip_ids: list of clip identifiers
#   paths: list of file paths
# ------------------------------------------------------
def pad_collate(batch):
    xs, ys, clip_ids, paths = zip(*batch)
    B = len(xs)
    C, F = xs[0].shape[:2]
    Tmax = max(x.shape[-1] for x in xs)  # maximum time dimension in the batch
    padded = torch.zeros(B, C, F, Tmax, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        T = x.shape[-1]
        padded[i, :, :, :T] = x
    targets = torch.stack(ys).float()
    return padded, targets, list(clip_ids), list(paths)

# ------------------------------------------------------
# Per-example z-score normalization for mel spectrograms
# Works with both NumPy arrays and Torch tensors.
# Normalizes each example independently: (x - mean) / std
# Helps reduce variance across examples with different loudness levels.
# ------------------------------------------------------
def per_example_zscore(x: np.ndarray | torch.Tensor, eps: float = 1e-6):
    if isinstance(x, np.ndarray):
        mean = x.mean(axis=(1,2), keepdims=True)
        std  = x.std(axis=(1,2), keepdims=True).clip(min=eps)
        return (x - mean) / std
    # torch.Tensor path
    mean = x.mean(dim=(1,2), keepdim=True)
    std  = x.std(dim=(1,2), keepdim=True).clamp_min(eps)
    return (x - mean) / std


