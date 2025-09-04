import argparse, csv
from pathlib import Path

IRMAS_LABELS = ["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]  # IRMAS train folders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to IRMAS-TrainingData")
    ap.add_argument("--out", type=Path, required=True, help="CSV to write")
    args = ap.parse_args()

    rows = []
    for lab in IRMAS_LABELS:
        for p in (args.root / lab).rglob("*.wav"):
            rows.append([str(p.resolve()), lab])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()