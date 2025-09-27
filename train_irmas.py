#!/usr/bin/env python3
"""
Train CNNVarTime on precomputed mel-dB tensors (.npy) listed in a manifest.

Assumptions:
- Manifest CSV has columns: filepath,label
- Each .npy is shaped (2, 128, T) float32, where T≈~300 for 3s @ 10ms hop
- Checkpoints saved under saved_weights/irmas_pretrain/
"""

from __future__ import annotations
from pathlib import Path
import argparse, math, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from CNN import CNNVarTime

# ------------------------
# Repro & device handling
# ------------------------
def setup_seed(seed: int = 1337):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def pick_device() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

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

# ------------------------
# Train/val
# ------------------------
# TODO: remove as included in train.ipynb
# def train(args):
#     setup_seed(args.seed)
#     device = pick_device()
#     print(f"Device: {device}")

#     # Labels once
#     tmp = SimpleMelNpyDataset(args.manifest, train=True, per_example_norm=True)
#     label_to_idx = tmp.label_to_idx
#     num_classes = len(label_to_idx)
#     print("Classes:", label_to_idx)

#     # Dataset & split
#     full = SimpleMelNpyDataset(args.manifest, label_to_idx=label_to_idx, train=True, per_example_norm=True)
#     N = len(full); n_val = int(round(N * args.val_frac)); n_train = N - n_val
#     train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(1337))
#     val_ds.dataset.train = False  # no aug anyway

#     pin_mem = (device == "cuda")
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=pin_mem, collate_fn=pad_time_axes)
#     val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
#                               num_workers=args.num_workers, pin_memory=pin_mem, collate_fn=pad_time_axes)

#     model = CNNVarTime(in_ch=2, num_classes=num_classes, p_drop=args.dropout).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
#     crit = nn.CrossEntropyLoss()

#     # Optional resume
#     if args.resume:
#         ckpt = torch.load(args.resume, map_location=device)
#         model.load_state_dict(ckpt["model_state"])
#         if args.resume_all and "opt_state" in ckpt and "sched_state" in ckpt:
#             opt.load_state_dict(ckpt["opt_state"])
#             sched.load_state_dict(ckpt["sched_state"])
#         print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch','?')})")

#     ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
#     best_val = 0.0; patience = args.patience; no_improve = 0

#     use_cuda_amp = (device == "cuda")
#     use_mps_amp  = (device == "mps")
#     scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

#     for epoch in range(1, args.epochs+1):
#         t0 = time.time()
#         # ---- train ----
#         model.train()
#         tr_loss = tr_acc = nb = 0
#         for X, y, _ in train_loader:
#             X = X.to(device, non_blocking=pin_mem); y = y.to(device, non_blocking=pin_mem)
#             opt.zero_grad(set_to_none=True)
#             if use_cuda_amp:
#                 with torch.cuda.amp.autocast():
#                     logits = model(X); loss = crit(logits, y)
#                 scaler.scale(loss).backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                 scaler.step(opt); scaler.update()
#             else:
#                 if use_mps_amp:
#                     with torch.autocast(device_type="mps", dtype=torch.float16):
#                         logits = model(X); loss = crit(logits, y)
#                 else:
#                     logits = model(X); loss = crit(logits, y)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                 opt.step()
#             tr_loss += loss.item(); tr_acc += (logits.argmax(1)==y).float().mean().item(); nb += 1
#         tr_loss /= max(1, nb); tr_acc /= max(1, nb)

#         # ---- val ----
#         model.eval()
#         va_loss = va_acc = nb = 0
#         with torch.no_grad():
#             for X, y, _ in val_loader:
#                 X = X.to(device, non_blocking=pin_mem); y = y.to(device, non_blocking=pin_mem)
#                 if use_cuda_amp:
#                     with torch.cuda.amp.autocast():
#                         logits = model(X); loss = crit(logits, y)
#                 else:
#                     if use_mps_amp:
#                         with torch.autocast(device_type="mps", dtype=torch.float16):
#                             logits = model(X); loss = crit(logits, y)
#                     else:
#                         logits = model(X); loss = crit(logits, y)
#                 va_loss += loss.item(); va_acc += (logits.argmax(1)==y).float().mean().item(); nb += 1
#         va_loss /= max(1, nb); va_acc /= max(1, nb)
#         sched.step()

#         print(f"[{epoch:03d}/{args.epochs}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | time {time.time()-t0:.1f}s")

#         # save last
#         torch.save({"epoch": epoch, "model_state": model.state_dict(),
#                     "opt_state": opt.state_dict(), "sched_state": sched.state_dict(),
#                     "label_to_idx": label_to_idx},
#                    ckpt_dir / "last.pt")

#         # early stopping on val-acc
#         if va_acc > best_val + 1e-6:
#             best_val = va_acc; no_improve = 0
#             torch.save({"epoch": epoch, "model_state": model.state_dict(),
#                         "opt_state": opt.state_dict(), "sched_state": sched.state_dict(),
#                         "label_to_idx": label_to_idx},
#                        ckpt_dir / "best_val_acc.pt")
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 print(f"Early stopping at epoch {epoch}. Best val acc {best_val:.4f}")
#                 break

#     print(f"Best val acc: {best_val:.4f}")

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/irmas_train_mels.csv")
    p.add_argument("--ckpt_dir", default="saved_weights/irmas_pretrain")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)  # on macOS try 0–2 if issues
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to warm-start model_state")
    p.add_argument("--resume_all", action="store_true", help="Also resume optimizer/scheduler states")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)