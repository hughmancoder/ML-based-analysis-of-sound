PY=python
# Data paths
# WAV=data/interim/Film1.wav
# PNG=data/processed/specs/Film1.png
WAV = data/Strings/mandolin/mandolin_A3_very-long_forte_tremolo.mp3
PNG = data/processed/mandolin_A3_very-long_forte_tremolo.png

.PHONY: venv install extract specs train infer clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

# Convert a WAV into a spectrogram image (mel, 3-channel)
specs:
	$(PY) scripts/to_spectrogram.py $(WAV) $(PNG) \
		--spec mel --mode stereo3 --sr 22050 \
		--win_ms 30 --hop_ms 10 --window boxcar \
		--n_mels 128 --img_size 224

# extract:
# 	$(PY) scripts/extract_audio.py data/raw/Film1.mp4 --out_dir data/interim --sr 22050

# clean:
# 	rm -rf checkpoints *.png