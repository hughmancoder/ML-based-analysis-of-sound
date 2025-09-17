# Structure (recommended)

project/
├─ data/
│  ├─ audio/
│  │  ├─ IRMAS-TrainingData/            # original IRMAS train (as downloaded)
│  │  └─ chinese_instruments/           # your curated set
│  │     ├─ gong/                       # per-class folders (single predominant label)
│  │     ├─ erhu/
│  │     └─ dizi/
│  ├─ mels/                             # generated features live here
│  │  ├─ IRMAS/
│  │  │  ├─ Train/<class>/*.png
│  │  │  └─ Val/<class>/*.png
│  │  └─ Chinese/
│  │     ├─ Train/<class>/*.png
│  │     └─ Val/<class>/*.png
│  └─ manifests/
│     ├─ irmas_train.csv
│     ├─ irmas_val.csv
│     ├─ chinese_train.csv
│     └─ chinese_val.csv
├─ tools/
│  └─ build_mels.py                     # script below
└─ train/
   └─ dataset_mel.py                    # tiny Dataset class (below)

## Structure suggestions

[template example](https://github.com/victoresque/pytorch-template)

## Dataset (IRMAS)

[Data set can be found here](https://zenodo.org/records/1290750#.WzCwSRyxXMU)
Specs: 44.1 kHz, 3.0 s 16 bit stereo WAV

## Focus

4 instruments: gong, erhu, dizi, pipa
- Extract audio from 3 target films.
- Segment into ~30 ms windows 