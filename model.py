import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False):
    # Conv → BatchNorm → ReLU (standard modern recipe)
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))  # halve image size
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = conv_block(channels, channels)

    def forward(self, x):
        # Skip connection: add original input to learned output
        return x + self.conv2(self.conv1(x))


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()

        self.stem   = conv_block(3, 64)                   # 3×32×32  → 64×32×32
        self.stage1 = conv_block(64, 128, pool=True)      # 64×32×32 → 128×16×16
        self.res1   = ResidualBlock(128)                   # 128×16×16 (skip)
        self.stage2 = conv_block(128, 256, pool=True)     # 128×16×16 → 256×8×8
        self.stage3 = conv_block(256, 512, pool=True)     # 256×8×8  → 512×4×4
        self.res2   = ResidualBlock(512)                   # 512×4×4  (skip)

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # 512×4×4 → 512×1×1
            nn.Flatten(),             # 512×1×1 → 512
            nn.Dropout(dropout),      # randomly zero 20% during training
            nn.Linear(512, num_classes),  # 512 → 10 class scores
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(self.stage1(x))
        x = self.stage2(x)
        x = self.res2(self.stage3(x))
        return self.head(x)
