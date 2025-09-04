import torch
import torch.nn as nn
# TODO: 
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=0 if k==2 else 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x): return self.block(x)

class CNNBaseline(nn.Module):
    """
    4 conv layers: (4x4)->8, (3x3)->16, (2x2)->32, (2x2)->64
    MaxPool(2x2) after each; FC 500 -> num_classes
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch,   8, k=4),   # -> downsample x2
            ConvBlock(8,      16, k=3),   # -> x4
            ConvBlock(16,     32, k=2),   # -> x8
            ConvBlock(32,     64, k=2),   # -> x16
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64* (128//16) * (300//16), 500)  # H=128, W≈300 for 3s@10ms hop
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        # x: (B,C,H,W)
        z = self.features(x)
        z = torch.flatten(z, 1)
        z = self.dropout(self.relu(self.fc1(z)))
        return self.fc2(z)