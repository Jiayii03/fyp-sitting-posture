import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  # Hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),  # Hidden layer 2
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.model(x)