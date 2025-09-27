IRMAS_DIR := data/audio/IRMAS/IRMAS-TrainingData
CHINESE_DIR := data/audio/chinese_instruments
MANIFESTS := data/manifests
MEL_IRMAS_IMG_ROOT := data/mels/irmas/
MEL_IMG_MANIFEST := $(MANIFESTS)/irmas_mels.csv
CACHE := .cache

SR := 44100
DUR := 3.0
N_MELS := 128
WIN_MS := 30.0
HOP_MS := 10.0
BATCH := 64
WORKERS := 8


.PHONY: manifests specs_irmas specs_chinese train_irmas clean_cache

# Data Generation
generate_gong_dataset:
	python data/scripts/generate_data_from_json.py --input data/audio/chinese_instruments/sources/gong.json

generate_dizi_dataset:
	python data/scripts/generate_data_from_json.py --input data/audio/chinese_instruments/sources/dizi.json

generate_guzheng_dataset:
	python data/scripts/generate_data_from_json.py --input data/audio/chinese_instruments/sources/guzheng.json

# NOTE: use with caution
generate_all_datasets: 
	generate_gong_dataset
	generate_dizi_dataset

summarise_data:
	python data/scripts/summarise_data.py --root $(CHINESE_DIR) 

# train manifests
manifests:
	python data/scripts/generate_train_manifests.py \
	--irmas_dir $(IRMAS_DIR) \
	--chinese_dir $(CHINESE_DIR) \
	--out_dir $(MANIFESTS)

# generates .npy mel spectrograms from IRMAS train_datasset
generate_irmas_train_spectrograms: manifests
	python data/scripts/precache_mels.py \
	  --manifest_csv $(MANIFESTS)/irmas_train.csv \
	  --cache_root $(CACHE)/mels_irmas \
	  --mel_manifest_out $(MANIFESTS)/irmas_train_mels.csv \
	  --batch_size $(BATCH) --workers $(WORKERS) \
	  --sr $(SR) --dur $(DUR) --n_mels $(N_MELS) \
	  --win_ms $(WIN_MS) --hop_ms $(HOP_MS)

# test manifests

# IRMAS_TEST_DIR := data/audio/IRMAS/IRMAS-TestingData-Part1

# IRMAS_TEST_MANIFEST      := $(MANIFESTS)/irmas_test.csv
# IRMAS_TEST_MELS_WINDOWS  := $(MANIFESTS)/irmas_test_mels_windows.csv

# generate_irmas_test_manifest:
# 	python data/scripts/generate_irmas_test_manifest.py \
# 	  --irmas_test_dir $(IRMAS_TEST_DIR) \
# 	  --out_csv $(IRMAS_TEST_MANIFEST)

# generate_irmas_test_spectrograms: generate_irmas_test_manifest
# 	python data/scripts/precache_mels_windows.py \
# 	  --test_manifest_csv $(IRMAS_TEST_MANIFEST) \
# 	  --cache_root $(CACHE)/mels_irmas \
# 	  --mel_manifest_out $(IRMAS_TEST_MELS_WINDOWS) \
# 	  --sr $(SR) --dur $(DUR) --n_mels $(N_MELS) \
# 	  --win_ms $(WIN_MS) --hop_ms $(HOP_MS) \
# 	  --hop_s 1.5 \
# 	  --relative_to $(CACHE)/mels_irmas


clean_cache:
	rm -rf $(CACHE)/mels_irmas $(CACHE)/mels_chinese $(CACHE)/canonical $(CACHE)/video_tmp 
