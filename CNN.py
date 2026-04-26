import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, n_channels_in=12, n_filters_1=32, n_filters_2=16):
        super(CNNModel, self).__init__()
        
        # Block 1: Initial Convolutions
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(n_channels_in, n_filters_1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(n_filters_1, n_filters_1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Reduced Feature Maps
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(n_filters_2, n_filters_2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.flatten = nn.Flatten()
        # Based on your calculation: 16 filters * 8 height * 42 width
        self.fc = nn.LazyLinear(1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = CNNModel()

