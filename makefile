PY        ?= python
PY_SRC    := PYTHONPATH=src $(PY)

# Paths
DATA_DIR          := data
IRMAS_TRAIN_DIR   := $(DATA_DIR)/audio/IRMAS/IRMAS-TrainingData
IRMAS_TEST_DIR    := $(DATA_DIR)/audio/IRMAS/IRMAS-TestingData-Part1
CHINESE_INSTRUMENTS_DIR  := $(DATA_DIR)/audio/chinese_instruments
CHN_SOURCES_DIR   := $(CHINESE_INSTRUMENTS_DIR)/sources
CHN_TRAIN_DIR   := $(CHINESE_INSTRUMENTS_DIR)/train

CACHE_DIR         := .cache
IRMAS_MELS_DIR    := $(CACHE_DIR)/mels/irmas
CHINESE_MELS_DIR    := $(CACHE_DIR)/mels/chinese_instruments
IRMAS_TEST_MELS_DIR   := $(CACHE_DIR)/mels/irmas/test

# Manifests
MANIFEST_DIR      := $(DATA_DIR)/manifests
IRMAS_TRAIN_MANIFEST := $(MANIFEST_DIR)/irmas_train.csv
IRMAS_TRAIN_MELS_CSV := $(MANIFEST_DIR)/irmas_train_mels.csv
CHINESE_INSTRUMENTS_TRAIN_MELS_CSV := $(MANIFEST_DIR)/chinese_instruments_train_mels.csv
IRMAS_TEST_MELS_CSV  := $(MANIFEST_DIR)/irmas_test_mels.csv

# Audio/Mel Parameters
SR        := 44100
DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 1.5 # TODO: try 1.5, 3

# Compute/IO
BATCH     := 64
WORKERS   := 8

#  Manifests (train) 
manifests: 
	$(PY_SRC) -m scripts.generate_train_manifests \
	  --irmas_dir $(IRMAS_TRAIN_DIR) \
	  --chinese_dir $(CHINESE_INSTRUMENTS_DIR) \
	  --out_dir $(MANIFEST_DIR)

generate_irmas_train_mels: ## Generate train mel cache + manifest
	$(PY_SRC) -m scripts.generate_irmas_train_mels \
	  --irmas_train_dir $(IRMAS_TRAIN_DIR) \
	  --cache_root $(IRMAS_MELS_DIR)/train \
	  --mel_manifest_out $(IRMAS_TRAIN_MELS_CSV) \
	  --sr $(SR) --dur $(DUR) --n_mels $(N_MELS) \
	  --win_ms $(WIN_MS) --hop_ms $(HOP_MS)
	 
generate_irmas_test_mels: ## Generate test mel windows + manifest
	$(PY_SRC) -m scripts.generate_irmas_test_mels \
	  --input_dir "$(IRMAS_TEST_DIR)" \
	  --cache_root "$(IRMAS_TEST_MELS_DIR)" \
	  --manifest_out "$(IRMAS_TEST_MELS_CSV)" \
	  --dataset_name IRMAS \
	  --project_root "$(PROJECT_ROOT)" \
	  --sr $(SR) --dur $(DUR) --n_mels $(N_MELS) \
	  --win_ms $(WIN_MS) --hop_ms $(HOP_MS) \
	  --stride_s $(STRIDE_S)

generate_chinese_train_mels: 
	$(PY_SRC) -m scripts.generate_irmas_train_mels \
	  --irmas_train_dir $(CHN_TRAIN_DIR) \
	  --cache_root $(CHINESE_MELS_DIR) \
	  --mel_manifest_out $(CHINESE_INSTRUMENTS_TRAIN_MELS_CSV) \
	  --sr $(SR) --dur $(DUR) --n_mels $(N_MELS) \
	  --win_ms $(WIN_MS) --hop_ms $(HOP_MS)

# Chinese instruments generation
chinese_percussion:  
	$(PY_SRC) -m scripts.generate_data_from_json --input $(CHN_SOURCES_DIR)/percussion.json

chinese_dizi: ## Build Dizi dataset from JSON source
	$(PY_SRC) -m scripts.generate_data_from_json --input $(CHN_SOURCES_DIR)/dizi.json

chinese_guzheng:  ## Build Guzheng dataset from JSON source
	$(PY_SRC) -m scripts.generate_data_from_json --input $(CHN_SOURCES_DIR)/guzheng.json

chinese_suona:  ## Build Suona dataset from JSON source
	$(PY_SRC) -m scripts.generate_data_from_json --input $(CHN_SOURCES_DIR)/suona.json

chinese_all: chinese_percussion chinese_dizi chinese_guzheng chinese_suona ## Build all Chinese datasets

chinese_summary: ## Summarise Chinese dataset directory
	$(PY_SRC) -m scripts.summarise_data --root $(CHN_TRAIN_DIR)

IRMAS_summary: ## Summarise Chinese dataset directory
	$(PY_SRC) -m scripts.summarise_data --root $(IRMAS_TRAIN_DIR)

clean_cache: ## Remove cached mel/spec/tmp data
	rm -rf $(CACHE_DIR)/mels $(CACHE_DIR)/mels_chinese $(CACHE_DIR)/canonical $(CACHE_DIR)/video_tmp
