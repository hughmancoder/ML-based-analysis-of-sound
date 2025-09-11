PY ?= python

DATA_ROOT := data
AUDIO_ROOT := $(DATA_ROOT)
PREP_ROOT  := $(DATA_ROOT)/standardized_wav
SPEC_ROOT  := $(DATA_ROOT)/mel_spectrogram

SR_AUDIO   := 44100
CLIP_SEC   := 3.0

SR         ?= 44100
DURATION_S ?= 3.0
WIN_MS     ?= 30
HOP_MS     ?= 10
N_MELS     ?= 128
FMIN       ?= 30
IMG_SIZE   ?= 224
SAVE_NPY   ?=

.PHONY: install preprocess_all specs

install:
	. .venv/bin/activate && pip install -r requirements.txt

preprocess_all:
	$(PY) scripts/standardize_wav.py "$(AUDIO_ROOT)" "$(PREP_ROOT)" \
		--sr $(SR_AUDIO) --duration_s $(CLIP_SEC) --recursive --verbose

specs: preprocess_all
	$(PY) scripts/to_spectrogram.py "$(PREP_ROOT)" "$(SPEC_ROOT)" \
		--mode stereo3 --sr $(SR) --duration_s $(DURATION_S) \
		--win_ms $(WIN_MS) --hop_ms $(HOP_MS) \
		--n_mels $(N_MELS) --fmin $(FMIN) --img_size $(IMG_SIZE) \
		--recursive $(SAVE_NPY) --verbose

extract_data:
	python data/scripts/video_to_wav_clips.py --json data/manifests/chinese_instruments.json
