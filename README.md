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

or use the provided `Makefile`:

### 2. Running project

refer to the make file for command lines

```bash
# 3) Generate spectrograms for all files (recursively):
make specs            # stereo3 by default (L, R, Mean -> 3-channel PNGs)

#    or strictly mono:
make specs_mono

# 4) One file only (uses FILENAME in Makefile):
make generate_one
```

## Goals

- Extract audio from 3 target films.
- Segment into ~30 ms windows (quasi-stationary).
- Generate FFT/Mel spectrograms (square/boxcar window baseline; Hann optional)
- VGG-based CNN (transfer learning) for 11 instrument classes
- Prototype spectrogram generator + full train/infer pipeline

## Libraries

[librosa](https://librosa.org/doc/latest/index.html)  for audio processing

## Resources

[paper respository](https://github.com/dhing1024/cs230-instrument-audio-ai)

## Structure suggestions

[template example](https://github.com/victoresque/pytorch-template)

## Extensions

- [ ] Mel + Δ + ΔΔ 3-channel spectrograms