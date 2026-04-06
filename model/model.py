import torch
import torch.nn as nn


class HealthcareModel(nn.Module):
    """
    Flexible MLP for healthcare classification.
    Supports any number of input features and output classes.
    """

    def __init__(self, input_size: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)
