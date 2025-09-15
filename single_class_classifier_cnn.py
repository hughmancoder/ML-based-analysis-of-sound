import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NCHANNELS = 2
NCLASSES = 11 # (IRMAS)
SAMPLE_SHAPE = (128, 259)  # (H, W) NOTE: modify to suit generated mel spectrogram