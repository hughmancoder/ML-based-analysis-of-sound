# Structure (recommended)

## Structure suggestions

[template example](https://github.com/victoresque/pytorch-template)

## Dataset (IRMAS)

[Data set can be found here](https://zenodo.org/records/1290750#.WzCwSRyxXMU)
Specs: 44.1 kHz, 3.0 s 16 bit stereo WAV

## Data Generation

we generate our labelled polyphonic dataset from a script which reads from a JSON file containing metadata about the audio/video sources to be used. 
This script automates our data extraction, segmentation, and labeling process
refer to [FORMAT.md](data/audio/chinese_instruments/sources/FORMAT.md) for JSON structure

## Focus

4 instruments: gong, erhu, dizi, pipa

- Extract audio from 3 target films
- Segment into ~30 ms windows 

## CNN output

Sigmoid for multi-label classification (last layer)

## Train Log

### Train train_irmas.py

model: src/models/CNNVarTime.py
Best val accuracy: 0.5632 
Best test set accuracy: 

## Checkpoint

[001/30] train 1.6473/0.4288 | val 1.4015/0.5610 | time 19.3s

## Other notes

- [ ] Use BCE loss function for multi-label classification
- [ ] be modeled as a polyphonic, multi-label classification problem, not a simple multiclass choice.
- [ ] Make model multi-label
- [ ] Improve dataset labels

## Datasets (Alternatives)

NSynth Dataset 2017, Google Brain, Deep Mind, Magenta
The NSynth dataset is highly useful for pre-training the core CNN feature extractor. It contains 305,979 one-shot musical notes from over 1,000 instruments, annotated by pitch, timbre, and envelope. Pre-training on NSynth allows the model to learn general, foundational concepts of timbre and pitch perception across diverse instrument families (e.g., brass, strings, woodwinds) before being introduced to the unique characteristics of Chinese instruments

https://magenta.withgoogle.com/datasets/nsynth

CH Music Dataset (1.1 GB)
ChMusic: A Traditional Chinese Music Dataset for Evaluation of Instrument Recognition
https://dl.acm.org/doi/10.1145/3490322.3490351#bibliography

Source: https://opendatalab.com/OpenDataLab/ChMusic/cli/main 

Repo: https://github.com/haoranweiutd/chmusic