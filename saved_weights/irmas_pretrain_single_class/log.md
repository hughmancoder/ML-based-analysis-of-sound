# Pretraining on IRMAS

Include following details
Parameter
Result
Model weight source
F1 Score

## CNNVarTime Training

Change the train number on each run

`TRAIN_NUMBER = "train_2" `

### Train 1

Preprocessing parameters
SR        := 44100
DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 3

EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.15
DROPOUT      = 0.3
PATIENCE     = 20
NUM_WORKERS  = 0 

epochs run: 137
validation_acuracy: 0.7197
model_weights `saved_weights/irmas_pretrain_single_class/train_1/best_val_acc_0.72.pt`


### Train 2

preprocessing_parameters
SR        := 44100
DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 3

EPOCHS       = 200
BATCH_SIZE   = 32
LR           = 3e-4


model_weights `saved_weights/irmas_pretrain_single_class/train_2`
validation_accuracy: 66.5%
Precision (0.66)
Recall (0.63)

### Train_3

SR        := 44100
DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 1.5


# train_6
Batch = 128
L5 = 0.001
Dropout = 0.3

75%

