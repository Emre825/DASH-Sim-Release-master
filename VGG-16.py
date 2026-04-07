"""
VGG-16: Very Deep Convolutional Network - PyTorch Implementation
Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
Authors: Karen Simonyan, Andrew Zisserman (2014)
"""

import torch
import torch.nn as nn

# input: 224x224
class VGG16(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, use_batch_norm=False):
        super().__init__()

        def conv_block(in_c, out_c, count):
            layers = []
            for _ in range(count):
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
                in_c = out_c
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        self.features = nn.Sequential(
            *conv_block(in_channels, 64, 2),    # -> 112x112x64
            *conv_block(64, 128, 2),            # -> 56x56x128
            *conv_block(128, 256, 3),           # -> 28x28x256
            *conv_block(256, 512, 3),           # -> 14x14x512
            *conv_block(512, 512, 3),           # -> 7x7x512
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)