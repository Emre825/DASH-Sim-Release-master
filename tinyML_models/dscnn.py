import torch
import torch.nn as nn
import torch.nn.functional as F

class DSCNNModel(nn.Module):
    def __init__(self, input_shape=(3, 25, 5), num_classes=10, weight_decay=1e-4):
        super(DSCNNModel, self).__init__()
        filters = 64

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=filters, kernel_size=(10, 4), stride=(2, 2), padding=(5, 2))

        self.depthwise_conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=filters)
        self.conv2_1x1_1 = nn.Conv2d(filters, filters, kernel_size=(1, 1))

        self.depthwise_conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=filters)
        self.conv2_1x1_2 = nn.Conv2d(filters, filters, kernel_size=(1, 1))

        self.depthwise_conv3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=filters)
        self.conv3_1x1 = nn.Conv2d(filters, filters, kernel_size=(1, 1))

        self.depthwise_conv4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=filters)
        self.conv4_1x1 = nn.Conv2d(filters, filters, kernel_size=(1, 1))

        self.bn1 = nn.BatchNorm2d(filters)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.pool = nn.AvgPool2d(kernel_size=(input_shape[1] // 2, input_shape[2] // 2))
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn1(self.depthwise_conv1(x)))
        x = F.relu(self.bn1(self.conv2_1x1_1(x)))

        x = F.relu(self.bn1(self.depthwise_conv2(x)))
        x = F.relu(self.bn1(self.conv2_1x1_2(x)))

        x = F.relu(self.bn1(self.depthwise_conv3(x)))
        x = F.relu(self.bn1(self.conv3_1x1(x)))

        x = F.relu(self.bn1(self.depthwise_conv4(x)))
        x = F.relu(self.bn1(self.conv4_1x1(x)))

        x = self.dropout2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

