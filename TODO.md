# TODO

## Data extraction scripts

- [x] Convert to spectrograms
- [x] Extraction script for IRMAS wav from video files . JSON format should read filepath
- [x] Add to also get data from local directory (windows/mac and it will append it to the manifest csv) (if array is empty take the whole video for the sample)
- [ ] ~700 samples for each instrument
- [x] Make manifests work for local paths instead of absolute paths

## Preprocessing

- [x] Summarise dataset labels script
- [x] Generate mel spectrogram pipeline for IRMAS 
- [ ] Ensure the train spectrograms are 301 samples wide too

## Training

- [x] Setup single instrument classification CNN
- [ ] Pretrain on IRMAS
- [ ] Finetune CNN to classify 4 instruments
- [ ] Fix train.ipynb and classification and create a ML branch
- [ ] try setting parameter collate_fn=pad_collate,     # required for variable-length T for both train and test

## Testing and Evaluation

- Train is single-label; test is multi-label.
- At inference,  model should output independent probabilities per class (sigmoid), and you threshold or rank them
- [x] Create overlapping clip windows for test set with 3s windows until end and aggregate window predictions. This becomes the clip prediction.
- [x] make test manifest for test windows (each row is a window, with start and end time)
- [x] Fix test window size (currently 201)
- [ ] Evaluate on trainset

- 
Idea:
```
mel_path,label_multi,orig_path,start_s,dur_s,n_frames
/path/to/cache/<label_dir>/<filehash>__...npy, 01000100000, /abs/path.wav, 0.0, 3.0, 301
```

## Other

- [x] Update README with new commands
- [ ] Resolve TODOS

## Team Instructions

Install IRMAS datasets at following locations

data/audio/IRMAS/IRMAS-TestingData-Part1
data/audio/IRMAS/IRMAS-TrainingData

Run preprocessing pipeline
make manifests
make generate_irmas_train_mels
generate_irmas_test_mels

practice trainign neural network in train.ipynb
