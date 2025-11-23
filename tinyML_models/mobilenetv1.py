import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, depthwise_initializer=nn.init.kaiming_normal_, pointwise_initializer=nn.init.kaiming_normal_):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        depthwise_initializer(self.depthwise.weight)
        pointwise_initializer(self.pointwise.weight)
        
    def forward(self, x):
        x = F.relu(self.bn_depthwise(self.depthwise(x)))
        x = F.relu(self.bn_pointwise(self.pointwise(x)))
        return x

class Mobilenetv1(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), num_classes=2, num_filters=8):
        super(Mobilenetv1, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], num_filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.layer2 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=1)
        
        num_filters *= 2
        self.layer3 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        
        num_filters *= 2
        self.layer4 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        
        self.layer5 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        
        num_filters *= 2
        self.layer6 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        
        self.layer7 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        
        num_filters *= 2
        self.layers_8_to_12 = nn.Sequential(
            DepthwiseSeparableConv(num_filters, num_filters, stride=1),
            DepthwiseSeparableConv(num_filters, num_filters, stride=1),
            DepthwiseSeparableConv(num_filters, num_filters, stride=1),
            DepthwiseSeparableConv(num_filters, num_filters, stride=1),
            DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        )
        
        self.layer13 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        
        num_filters *= 2
        self.layer14 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layers_8_to_12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
