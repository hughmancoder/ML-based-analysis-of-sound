## TODO

## Data extraction scripts

- [x] Convert to spectrograms
- [x] Extraction script for IRMAS wav from video files . JSON format should read filepath
- [x] Add to also get data from local directory (windows/mac and it will append it to the manifest csv) (if array is empty take the whole video for the sample)
- [ ] ~700 samples for each instrument

## Preprocessing

- [ ] Normalise Spectrograms
- [ ] Ensure hanning widnow parameter is applied
- [ ] Summarise dataset labels script

## Training

- [ ] Add more sound files to the dataset
- [ ] Setup single instrument classification CNN
- [ ] Train CNN to classify 4 instruments
- [ ] Validation at load time


## Other

- [x] Update README with new commands
