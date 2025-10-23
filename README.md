# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

[Project Board](https://github.com/users/hughmancoder/projects/4)

## Setup

Install prequisites on your machine

`git, python3, pip, make`

### Setup environment

```bash
# Create virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate   

# On Windows (cmd.exe)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Activate environment (venv) on every new terminal 

### Run the project

refer to the make file for command lines

```bash
make chinese_all # generates dataset from video files
```

## Workflow

Install IRMAS datasets at following locations

data/audio/IRMAS/IRMAS-TestingData-Part1
data/audio/IRMAS/IRMAS-TrainingData

Run preprocessing pipeline to generate the dataset

```bash
make manifests
make generate_irmas_train_mels
generate_irmas_test_mels
```

