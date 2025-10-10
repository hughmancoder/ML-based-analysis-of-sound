

# Canonical IRMAS class order (must match training!)
IRMAS_CLASSES = [
    "cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"
]

# LABEL_TO_IDX_IRMAS = {
#     'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10
# }

# IDX_TO_LABEL_IRMAS = {v:k for k,v in LABEL_TO_IDX_IRMAS.items()}

# import random
# import torch
# import numpy as np
# import pandas as pd

# LABEL_TO_IDX_CN = {'guzheng':0, 'suona':1, 'dizi':2, 'gong':3}
# IDX_TO_LABEL_CN = {v:k for k,v in LABEL_TO_IDX_CN.items()}

# def setup_seed(seed: int = 1337):
#     torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# def load_npy(path: str) -> np.ndarray:
#     """Load a cached mel tensor saved by precache scripts."""
#     return np.load(path)

# def decode_label_bits(bits: str, classes) -> str:
#     """Turn a 0/1 multi-hot string into comma-separated class labels."""
#     clean = "".join(ch for ch in str(bits).strip() if ch in "01")
#     if len(clean) < len(classes):
#         clean = clean.ljust(len(classes), "0")
#     else:
#         clean = clean[:len(classes)]
#     labels = [cls for cls, flag in zip(classes, clean) if flag == "1"]
#     return ", ".join(labels) if labels else "none"

