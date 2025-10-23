# TODO

## Data extraction scripts

- [x] Convert to spectrograms
- [x] Extraction script for IRMAS wav from video files . JSON format should read filepath
- [x] Add to also get data from local directory (windows/mac and it will append it to the manifest csv) (if array is empty take the whole video for the sample)
- [x] ~700 samples for each instrument
- [x] Make manifests work for local paths instead of absolute paths

## Preprocessing

- [x] Summarise dataset labels script
- [x] Generate mel spectrogram pipeline for IRMAS 
- [x] Ensure the train spectrograms are 301 samples wide too
- [x] Confusion matrixx`
- [x] Remove one hot encoded labels in favour of a more flexible scheme (have a think about this)
- [x] Use generate IRMAS_test_mels.py to generate chinese data
- [ ] Make generated instrument manifest to be adjacent to generated files

## Training

- [x] Setup single instruøment classification CNN
- [x] Pretrain on IRMAS
- [ ] Finetune CNN to classify 4 instruments
- [ ] try setting parameter collate_fn=pad_collate,     # required for variable-length T for both train and test
- [ ] Fine tune CNN to multilabel classification
- [ ] Multiclass Loss: switch to BCEWithLogitsLoss (not CrossEntropy).
- [ ] Generated mels should go in the data folder not .cache
  
## Testing and Evaluation

- Train is single-label; test is multi-label.
- At inference,  model should output independent probabilities per class (sigmoid), and you threshold or rank them
- [x] Create overlapping clip windows for test set with 3s windows until end and aggregate window predictions. This becomes the clip prediction.
- [x] make test manifest for test windows (each row is a window, with start and end time)
- [x] Fix test window size (currently 201)
- [x] Evaluate on trainset
- [x] Improve evaluation on unseen data

## Bugs

- [ ] Fix Test single class and improt errors

- 
Idea:
```
mel_path,label_multi,orig_path,start_s,dur_s,n_frames
/path/to/cache/<label_dir>/<filehash>__...npy, 01000100000, /abs/path.wav, 0.0, 3.0, 301
```

## Other

- [x] Update README with new commands
- [ ] Resolve TODOS

