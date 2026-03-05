# -------------------------------
#  ResNet9 CNN Model for CIFAR-10
# -------------------------------

import torch
import torch.nn as nn
from einops import rearrange  # handy tool for reshaping tensors (not strictly needed here)

# -------------------------------
#  Helper: Convolutional Block
# -------------------------------
def conv_block(in_channels, out_channels, pool=False):
 
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))  # shrink HxW by half
    return nn.Sequential(*layers)

# -------------------------------
#  Residual Block (Skip Connections)
# -------------------------------
class ResidualBlock(nn.Module):
  
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels)   
        self.conv2 = conv_block(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))  # 'skip connection': add input to processed output

# -------------------------------
#  ResNet9 Architecture
# -------------------------------
class ResNet9(nn.Module):

    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()

        # First layers (stem)
        self.stem   = conv_block(3,   64)       # input image has 3 channels (RGB)
        self.stage1 = conv_block(64,  128, pool=True)
        self.res1   = ResidualBlock(128)
        self.stage2 = conv_block(128, 256, pool=True)
        self.stage3 = conv_block(256, 512, pool=True)
        self.res2   = ResidualBlock(512)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),   # shrink feature map to 1x1
            nn.Flatten(),               # flatten to vector
            nn.Dropout(dropout),        # randomly zero some neurons (regularization)
            nn.Linear(512, num_classes), # final output: 10 classes for CIFAR-10
        )

    # -------------------------------
    # Forward pass
    # -------------------------------
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.res1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.res2(x)
        x = self.head(x)
        return x                     # raw scores (logits) for each class

# -------------------------------
# Quick test to check model
# -------------------------------
if __name__ == '__main__':
    model = ResNet9()
    dummy = torch.randn(4, 3, 32, 32)  # pretend batch of 4 images
    print("Output shape:", model(dummy).shape)  # should be [4, 10]
    
    # Count trainable parameters
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n:,}")  # ~6.6 million
