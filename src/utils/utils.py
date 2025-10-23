from pathlib import Path
from src.classes import IRMAS_CLASSES, ALL_CLASSES

# def _label_from_txt_one_hot(txt_path: Path) -> str:
#     present = set()
#     if txt_path.exists():
#         for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
#             k = line.strip().lower()
#             if k in IRMAS_CLASSES:
#                 present.add(k)
#     # Preserve width and leading zeros
#     return "".join("1" if c in present else "0" for c in IRMAS_CLASSES)

def _label_from_txt(txt_path: Path) -> list[str]:
    """
    Reads IRMAS-style .txt file and returns a list of instrument labels, e.g. ['gel', 'pia'].
    """
    if not txt_path.exists():
        return []

    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    labels = []
    for ln in lines:
        # IRMAS txt lines often look like: 'gel 0.8' or 'gel'
        token = ln.split()[0]
        if token in ALL_CLASSES:
            labels.append(token)
        else:
            # Optional: normalize or warn
            print(f"[WARN] Unknown label '{token}' in {txt_path}")
    return labels


# def decode_label_bits(bits: str, classes) -> str:
#     """Turn a 0/1 multi-hot string into comma-separated class labels."""
#     clean = "".join(ch for ch in str(bits).strip() if ch in "01")
#     if len(clean) < len(classes):
#         clean = clean.ljust(len(classes), "0")
#     else:
#         clean = clean[:len(classes)]
#     labels = [cls for cls, flag in zip(classes, clean) if flag == "1"]
#     return ", ".join(labels) if labels else "none"



