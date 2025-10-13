# Training CNN

## Pretraining IRMAS

Include following details
Parameter
Result
Model weight source

## CNNVarTime

### Train 1

Junran

EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.15
DROPOUT      = 0.3
PATIENCE     = 20
NUM_WORKERS  = 0 

epochs run: 137
validation_acuracy: 72.0%
model_weights `saved_weights/irmas_pretrain_single_class/train_1/best_val_acc_0.72.pt`

### Train 2

preprocessing_parameters

EPOCHS       = 200
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.15
DROPOUT      = 0.5
PATIENCE     = 50
NUM_WORKERS  = 2

model_weights `saved_weights/irmas_pretrain_single_class/train_2`
validation_accuracy: 66.5%



