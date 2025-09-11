# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

## Setup

## How to run

*## Prequisites
Make sure the following are installed on your machine:
- git, python, pip, make

### 1. Setup environment

```bash
# Create virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate   

# On Windows: 
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

or use the provided `Makefile`:

### 2. Running project

refer to the make file for command lines

```bash
make extract_data # generates dataset from video files

```
## Libraries

[librosa](https://librosa.org/doc/latest/index.html)  for audio processing

## Resources

[paper respository](https://github.com/dhing1024/cs230-instrument-audio-ai)

## Structure suggestions

[template example](https://github.com/victoresque/pytorch-template)

## IRMAS

[Data set can be found here](https://zenodo.org/records/1290750#.WzCwSRyxXMU)
Specs: 44.1 kHz, 3.0 s 16 bit stereo WAV