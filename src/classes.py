
IRMAS_CLASSES = [
    "cel", "cla", "flu", "gac", "gel", "org",
    "pia", "sax", "tru", "vio", "voi"
]

CHINESE_CLASSES = [
    "guzheng", "suona", "dizi", "percussion"
]

# Unified list (update here when adding more sets)
ALL_CLASSES = IRMAS_CLASSES + CHINESE_CLASSES

# Convenient lookup dicts for dataloaders
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}