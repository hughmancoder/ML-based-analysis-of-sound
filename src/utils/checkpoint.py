# # src/utils/ckpt.py
# from __future__ import annotations
# from typing import Dict, List, Tuple, Any
# import torch

# def load_model_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     # prefer 'model_state' if present (your files use this)
#     for k in ("model_state", "model_state_dict", "state_dict", "model"):
#         if isinstance(ckpt, dict) and isinstance(ckpt.get(k), dict):
#             sd = ckpt[k]
#             break
#     else:
#         sd = ckpt if isinstance(ckpt, dict) else ckpt
#     # strip DDP prefix
#     return {k.replace("module.", "", 1): v for k, v in sd.items()}

# def class_names_from_ckpt(ckpt_path: str, fallback: List[str]) -> Tuple[List[str], Dict[str,int]]:
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     if isinstance(ckpt, dict) and isinstance(ckpt.get("label_to_idx"), dict):
#         l2i = ckpt["label_to_idx"]
#         # names in index order used during training
#         ordered = [name for name, idx in sorted(l2i.items(), key=lambda kv: kv[1])]
#         return ordered, l2i
#     # fallback to your provided order
#     return list(fallback), {n:i for i, n in enumerate(fallback)}

# train_order_names, _ = class_names_from_ckpt(RESUME_CKPT, fallback=list(IRMAS_CLASSES))
# test_dataset = IRMASTestWindowDataset(
#     manifest_csv=IRMAS_TEST_MANIFEST,
#     project_root=PROJECT_ROOT,
#     class_names=train_order_names,
#     per_example_norm=True,   # must match your train setting
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=None,
#     collate_fn=None,
# )