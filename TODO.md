## TODO

- [x] Convert to spectrograms
- [x] Extraction script for IRMAS wav from video files . JSON format should read filepath
- [ ] Add to also get data from local directory (windows/mac and it will append it to the manifest csv) (if array is empty take the whole video for the sample)
- [ ] Update README with new commands
- [ ] Normalise Spectrograms
- [ ] Ensure hanning widnow parameter is applied
- [x] Fix bug with 2 channel spectrograms settings png to black 
- [ ] Extracting audio from film files
- [ ] Add more sound files to the dataset
- [ ] Setup single instrument classification CNN
- [ ] Train CNN to classify 4 instruments

## Goals

- Extract audio from 3 target films.
- Segment into ~30 ms windows (quasi-stationary).
- Generate FFT/Mel spectrograms (square/boxcar window baseline; Hann optional)
- VGG-based CNN (transfer learning) for 11 instrument classes
- Prototype spectrogram generator + full train/infer pipeline

## Data format

Store both labels.txt (one per line) and labels.json (name→index). The text file is human-friendly; the JSON is machine-safe.

Validation at load time

Assert every row’s label exists in labels.json.

Assert label_idx == labels[label].