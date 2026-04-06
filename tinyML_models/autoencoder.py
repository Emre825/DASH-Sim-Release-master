import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDenseAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleDenseAutoEncoder, self).__init__()

        # Define each layer individually
        self.fc1 = nn.Linear(128, 128)
        
        self.fc2 = nn.Linear(128, 128)
        
        self.fc3 = nn.Linear(128, 128)
        
        self.fc4 = nn.Linear(128, 128)
        
        self.fc5 = nn.Linear(128, 8)
        
        self.fc6 = nn.Linear(8, 128)
        
        self.fc7 = nn.Linear(128, 128)
        
        self.fc8 = nn.Linear(128, 128)
        
        self.fc9 = nn.Linear(128, 128)
        
        self.output_layer = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.output_layer(x)
        return x



