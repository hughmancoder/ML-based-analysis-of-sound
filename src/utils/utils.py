
import numpy as np
import pandas as pd

# Canonical IRMAS class order (must match training!)
CLASSES = [
    "cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"
]

LABEL_TO_IDX_IRMAS = {
    'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10
}

IDX_TO_LABEL_IRMAS = {v:k for k,v in LABEL_TO_IDX_IRMAS.items()}

LABEL_TO_IDX_CN = {'guzheng':0, 'suona':1, 'dizi':2, 'gong':3}
IDX_TO_LABEL_CN = {v:k for k,v in LABEL_TO_IDX_CN.items()}

def load_npy(path: str) -> np.ndarray:
    """Load a cached mel tensor saved by precache scripts."""
    return np.load(path)

def decode_label_bits(bits: str, classes) -> str:
    """Turn a 0/1 multi-hot string into comma-separated class labels."""
    clean = "".join(ch for ch in str(bits).strip() if ch in "01")
    if len(clean) < len(classes):
        clean = clean.ljust(len(classes), "0")
    else:
        clean = clean[:len(classes)]
    labels = [cls for cls, flag in zip(classes, clean) if flag == "1"]
    return ", ".join(labels) if labels else "none"

def load_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Load a manifest CSV with consistent label handling.
    
    - Preserves leading zeros in label_multi.
    - Normalizes label_multi to exactly len(CLASSES) bits.
    - If a 'label' column exists, keeps it stringified.
    - Adds a human-readable decoded label column.
    """
    df = pd.read_csv(
        manifest_csv,
        dtype={"label_multi": "string"},  # keep leading zeros
        keep_default_na=False             # don't interpret "000..." as NaN
    )

    # If plain 'label' column exists, normalize it
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Clean and normalize label_multi to match CLASSES length
    if "label_multi" in df.columns:
        df["label_multi"] = (
            df["label_multi"].astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"[^01]", "", regex=True)
            .str.pad(len(CLASSES), side="right", fillchar="0")
            .str[:len(CLASSES)]
        )
        # Add decoded labels for convenience
        df["label"] = df["label_multi"].apply(lambda s: decode_label_bits(s, CLASSES))

    return df