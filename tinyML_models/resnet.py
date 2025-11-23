import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If input and output channels differ, adjust the identity path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # returns only x if stride = 1 and in_channels = out_channels, returns BN2D(Conv2D(x)) otherwise 
        out = F.relu(out)
        return out

class ResNetCIFAR10(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(ResNetCIFAR10, self).__init__()
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual stacks
        self.layer1 = self._make_layer(16, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=1, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=1, stride=2)
        
        # Uncomment the fourth stack if desired
        # self.layer4 = self._make_layer(128, num_blocks=1, stride=2)

        # Average Pooling and Fully Connected Layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes) # 64 represents the input features, which is found by CxHxW, which is 64 x 1 x 1 after adaptive average pool 2d. And num_classes represent output features.

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks): # since num_blocks is equal to 1, this is an empty range and the loop does not execute.
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out