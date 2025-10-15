# Result
Model weight source

## CNNVarTime
## CNNVarTime Training

Change the train number on each run

`TRAIN_NUMBER = "train_2" `

### Train 1

Junran
Preprocessing parameters  
SR        := 44100  
DUR       := 3.0  
N_MELS    := 128  
WIN_MS    := 30.0  
HOP_MS    := 10.0  
STRIDE_S  := 3  

EPOCHS       = 200  
BATCH_SIZE   = 128  
PATIENCE     = 20  
NUM_WORKERS  = 0  

epochs run: 137  
validation_acuracy: 72.0%  
validation_acuracy: 0.7197  
model_weights `saved_weights/irmas_pretrain_single_class/train_1/best_val_acc_0.72.pt`

---

### Train 2

preprocessing_parameters  
SR        := 44100  
DUR       := 3.0  
N_MELS    := 128  
WIN_MS    := 30.0  
HOP_MS    := 10.0  
STRIDE_S  := 3  

EPOCHS       = 200  
BATCH_SIZE   = 32  
NUM_WORKERS  = 2  

model_weights `saved_weights/irmas_pretrain_single_class/train_2`  
validation_accuracy: 66.5%  
Precision (0.66)  
Recall (0.63)

---

## Train 4

EPOCHS = 250  
BATCH_SIZE = 64  
LR = 5e-4  
WEIGHT_DECAY = 1.1e-3  
VAL_FRAC = 0.15  
DROPOUT = 0.5  
PATIENCE = 50  
NUM_WORKERS = 0  
SEED = 1337  

model_weights `saved_weights/irmas_pretrain_single_class/train_4`  
validation_accuracy: 69.1%

---

## Train 5

EPOCHS = 275  
BATCH_SIZE = 128  
LR = 1e-3  
WEIGHT_DECAY = 5e-4  
VAL_FRAC = 0.15  
DROPOUT = 0.4  
PATIENCE = 50  
NUM_WORKERS = 0  
SEED = 1337  

model_weights `saved_weights/irmas_pretrain_single_class/train_5`  
validation_accuracy: 70.9%

---

## Train 6

EPOCHS       = 500  
BATCH_SIZE   = 128  
LR           = 1e-3  
WEIGHT_DECAY = 1e-4  
VAL_FRAC     = 0.15  
DROPOUT      = 0.3  
PATIENCE     = 100  
NUM_WORKERS  = 0  

model_weights `saved_weights/irmas_pretrain_single_class/train_6`  
validation_accuracy: 74.7%  

SR        := 44100  
DUR       := 3.0  
N_MELS    := 128  
WIN_MS    := 30.0  
HOP_MS    := 10.0  
STRIDE_S  := 1.5
