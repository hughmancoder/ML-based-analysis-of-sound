# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

## Setup

## How to run

### 1. Setup environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running project

```
python scripts/extract_audio.py data/raw/Film1.mp4 --out_dir data/interim --sr 22050
```

## TODO

- [ ] Extracting audio from film files
- [x] Convert to spectrograms
- [ ] Add more sound files to the dataset
- [ ] Train CNN to classify 11 instruments


## Goals

- Extract audio from 3 target films.
- Segment into ~30 ms windows (quasi-stationary).
- Generate FFT/Mel spectrograms (square/boxcar window baseline; Hann optional)
- VGG-based CNN (transfer learning) for 11 instrument classes
- Prototype spectrogram generator + full train/infer pipeline
