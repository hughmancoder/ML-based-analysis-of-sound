from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Tuple, Optional
from src.utils.datasets import SingleClassMelNpyDataset
from src.models import CNNVarTime
from typing import List, Tuple, Dict


def seed_everything(seed: int = 1337):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device() -> str:
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"

def get_device():
    device = pick_device()  # "cuda" | "mps" | "cpu"
    use_cuda_amp = (device == "cuda")
    use_mps_amp = (device == "mps")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    pin_mem = (device == "cuda")
    return device, use_cuda_amp, use_mps_amp, scaler, pin_mem


def build_dataloaders(
    manifest_csv: str,
    val_frac: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = False,
    seed: int = 1337,
    label_to_idx: Optional[Dict[str,int]] = None,
):
    # Build label map (stable & reusable)
    if label_to_idx is None:
        tmp_ds = SingleClassMelNpyDataset(manifest_csv, per_example_norm=True)
        label_to_idx = tmp_ds.label_to_idx

    full_ds = SingleClassMelNpyDataset(
        manifest_csv,
        label_to_idx=label_to_idx,
        per_example_norm=True,
    )

    N = len(full_ds)
    n_val = int(round(N * val_frac))
    n_train = N - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=None
    )

    return train_loader, val_loader, label_to_idx


def build_model(num_classes: int, dropout: float = 0.5, in_ch: int = 2, device: str = "cpu"):
    model = CNNVarTime(in_ch=in_ch, num_classes=num_classes, p_drop=dropout).to(device)
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=pin_mem)
            y = y.to(device, non_blocking=pin_mem)

            if use_cuda_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(X)
                    loss = criterion(logits, y)
            else:
                if use_mps_amp:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        logits = model(X)
                        loss = criterion(logits, y)
                else:
                    logits = model(X)
                    loss = criterion(logits, y)

            loss_sum += float(loss.item()) * y.size(0)
            correct  += int((logits.argmax(1) == y).sum().item())
            total    += int(y.size(0))
    va_loss = loss_sum / max(1, total)
    va_acc  = correct / max(1, total)
    return va_loss, va_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    use_cuda_amp: bool,
    use_mps_amp: bool,
    pin_mem: bool
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=pin_mem)
        y = y.to(device, non_blocking=pin_mem)
        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_mps_amp:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    logits = model(X)
                    loss = criterion(logits, y)
            else:
                logits = model(X)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        loss_sum += float(loss.item()) * y.size(0)
        correct  += int((logits.argmax(1) == y).sum().item())
        total    += int(y.size(0))

    tr_loss = loss_sum / max(1, total)
    tr_acc  = correct / max(1, total)
    return tr_loss, tr_acc


# --------------------------- Checkpointing ---------------------------

def save_checkpoint(payload: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    current_label_to_idx: Optional[Dict[str,int]] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=torch.device(device))

    # label map consistency
    ckpt_label_to_idx = ckpt.get("label_to_idx")
    if current_label_to_idx is not None and ckpt_label_to_idx is not None:
        if ckpt_label_to_idx != current_label_to_idx:
            raise RuntimeError("label_to_idx mismatch between checkpoint and current dataset mapping.")

    model.load_state_dict(ckpt["model_state"], strict=True)

    if optimizer is not None and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    if scheduler is not None and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    if scaler is not None and "scaler_state" in ckpt and scaler.is_enabled():
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            # If device/AMP mode changed, scaler may be incompatible.
            pass

    return ckpt

def load_model_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = None
    if isinstance(ckpt, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if isinstance(ckpt.get(k), dict):
                sd = ckpt[k]; break
    if sd is None:
        sd = ckpt if isinstance(ckpt, dict) else ckpt
    # strip potential DDP prefix
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def class_names_from_ckpt(ckpt_path: str, fallback: List[str]) -> Tuple[List[str], Dict[str,int]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("label_to_idx"), dict):
        l2i = ckpt["label_to_idx"]
        ordered = [name for name, idx in sorted(l2i.items(), key=lambda kv: kv[1])]
        return ordered, l2i
    # Fallback only if absolutely necessary
    return list(fallback), {n:i for i, n in enumerate(fallback)}

# ------------------------------ Trainer -----------------------------

def train_loop(
    train_manifest: str,
    ckpt_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    dropout: float = 0.5,
    patience: int = 50,
    num_workers: int = 2,
    seed: int = 1337,
    resume_from: Optional[Path] = None,   # e.g., ckpt_dir/"last.pt" or ckpt_dir/"best_val_acc.pt"
    save_best_stamped: bool = True,       # also save best_eXXXX_accYY.YY.pt
) -> Dict[str, Any]:

    seed_everything(seed)
    device, use_cuda_amp, use_mps_amp, scaler, pin_mem = get_device()
    print("Device:", device)

    # Build data
    train_loader, val_loader, label_to_idx = build_dataloaders(
        train_manifest, val_frac, batch_size, num_workers, pin_mem, seed
    )
    print("train size:", len(train_loader.dataset), " val size:", len(val_loader.dataset))

    # Build model/optim/sched/criterion
    num_classes = len(label_to_idx)
    print("Classes:", label_to_idx)
    model = build_model(num_classes=num_classes, dropout=dropout, in_ch=2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Resume bookkeeping
    start_epoch = 1
    best_val_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if resume_from is not None and Path(resume_from).exists():
        print(f"Resuming from {resume_from}")
        ckpt = load_checkpoint(
            Path(resume_from), device, model, optimizer, scheduler, scaler, current_label_to_idx=label_to_idx
        )
        start_epoch   = int(ckpt.get("epoch", 0)) + 1
        history       = ckpt.get("history", history)
        best_val_acc  = float(ckpt.get("best_val_acc", best_val_acc))
        no_improve    = int(ckpt.get("no_improve", no_improve))
        print(f"Resumed at epoch {start_epoch} (best_val_acc={best_val_acc:.4f}, no_improve={no_improve})")
    else:
        print("Starting fresh.")

    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_cuda_amp, use_mps_amp, pin_mem
        )
        va_loss, va_acc = evaluate(
            model, val_loader, criterion, device, use_cuda_amp, use_mps_amp, pin_mem
        )

        scheduler.step()

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)

        dt = time.time() - t0
        print(f"[{epoch:03d}/{epochs}] train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | time {dt:.1f}s")

        # Save "last"
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "label_to_idx": label_to_idx,
            "history": history,
            "best_val_acc": best_val_acc,
            "no_improve": no_improve,
        }, ckpt_dir / "last.pt")

        # Save "best"
        if va_acc > best_val_acc + 1e-6:
            best_val_acc = va_acc
            no_improve = 0
            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "label_to_idx": label_to_idx,
                "history": history,
                "best_val_acc": best_val_acc,
                "no_improve": no_improve,
            }
            save_checkpoint(payload, ckpt_dir / "best_val_acc.pt")
            if save_best_stamped:
                stamped = ckpt_dir / f"best_epoch_{epoch:04d}_val_acc_{(va_acc*100):05.2f}.pt"
                save_checkpoint(payload, stamped)
            print(f"  ↑ new best val_acc: {best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val acc {best_val_acc:.4f}")
                break

    print("Best val acc:", best_val_acc)
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "label_to_idx": label_to_idx,
        "device": device,
    }