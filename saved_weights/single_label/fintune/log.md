# Fine-Tune Training CNN on chinese single class dataset

## Preprocessing parameters

DUR       := 3.0
N_MELS    := 128
WIN_MS    := 30.0
HOP_MS    := 10.0
STRIDE_S := 1.5 

## train_0 (Training from scratch on chinese dataset)

N = 15 labels
EPOCHS = 200
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.15
DROPOUT = 0.5
PATIENCE = 50
NUM_WORKERS = 2
SEED = 1337

99% train and validation accuracy on 4 classes for single class-label

## train_1 (Fine-tuning from IRMAS pretrained weights)

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_FRAC = 0.15
patience = 20

Early stopping at epoch 93. Best val acc 0.7592
Best val acc: 0.759

# train_6

