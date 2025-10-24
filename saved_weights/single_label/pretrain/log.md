# Pretraining on IRMAS

Include following details
Parameter
Result
Model weight source
F1 Score

## CNNVarTime Training

Change the train name
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

## Train 2

EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.5
VAL_FRAC = 0.15
Val accuracy = 0.725 

## Train 3

SR        := 44100
DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 1.5


## train_4

Batch = 128
LR = 0.001
Dropout = 0.3
Val accuracy 75% @ 84 epochs

