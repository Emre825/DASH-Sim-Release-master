"""
SqueezeNet v1.1: PyTorch Implementation
Paper: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
Authors: Iandola et al. (2016)
"""

import torch
import torch.nn as nn

class FireModule(nn.Module):
    """
    Fire Module: squeeze (1x1) -> expand (1x1 || 3x3) -> concat
    Output channels = e1x1 + e3x3
    """
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, s1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(s1x1, e1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand_1x1(s), self.expand_3x3(s)], dim=1)

# input: 224x224
class SqueezeNet(nn.Module):
    """
    SqueezeNet v1.1 — 2.4x less computation than v1.0, same accuracy.
    ~1.24M parameters, no fully connected layers.
    """
    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        # Fire modules: (in_channels, s1x1, e1x1, e3x3)
        self.fire2 = FireModule(64,  16,  64,  64)   # -> 128
        self.fire3 = FireModule(128, 16,  64,  64)   # -> 128
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire4 = FireModule(128, 32, 128, 128)   # -> 256
        self.fire5 = FireModule(256, 32, 128, 128)   # -> 256
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire6 = FireModule(256, 48, 192, 192)   # -> 384
        self.fire7 = FireModule(384, 48, 192, 192)   # -> 384
        self.fire8 = FireModule(384, 64, 256, 256)   # -> 512
        self.fire9 = FireModule(512, 64, 256, 256)   # -> 512

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool1(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool2(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)