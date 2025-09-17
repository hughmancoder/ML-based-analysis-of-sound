IRMAS_DIR := data/audio/IRMAS/IRMAS-TrainingData
CHINESE_DIR := data/audio/chinese_instruments
MANIFESTS := data/manifests
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

# NOTE: use with caution
generate_all_datasets: 
	generate_gong_dataset
	generate_dizi_dataset

summarise_data:
	python data/scripts/summarise_data.py --root $(CHINESE_DIR) 

# Preprocessing
manifests:
	python data/scripts/generate_manifests.py --irmas_dir $(IRMAS_DIR) --chinese_dir $(CHINESE_DIR) --out_dir $(MANIFESTS)

# generates mel-spectrograms from irmas
specs_irmas: manifests
	python data/scripts/precache_mels.py --manifest_csv $(MANIFESTS)/irmas_all.csv \
		--cache_root $(CACHE)/mels_irmas \
		--batch_size $(BATCH) --workers $(WORKERS) \
		--sr $(SR) --dur $(DUR) --n_mels $(N_MELS) --win_ms $(WIN_MS) --hop_ms $(HOP_MS)

specs_chinese: manifests
	python data/scripts/precache_mels.py --manifest_csv $(MANIFESTS)/chinese_all.csv \
		--cache_root $(CACHE)/mels_chinese \
		--batch_size $(BATCH) --workers $(WORKERS) \
		--sr $(SR) --dur $(DUR) --n_mels $(N_MELS) --win_ms $(WIN_MS) --hop_ms $(HOP_MS)

# train_irmas:
# 	python train/train_irmas.py --cache_root $(CACHE)/mels_irmas

clean_cache:
	rm -rf $(CACHE)/mels_irmas $(CACHE)/mels_chinese $(CACHE)/canonical $(CACHE)/video_tmp
