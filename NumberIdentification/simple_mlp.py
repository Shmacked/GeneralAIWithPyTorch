import torch.nn as nn
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_size=(28, 28), hidden_size=128, num_classes=10):
        super().__init__()
        _input_size = np.prod(input_size)
        
        self.main_structure = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.main_structure(x)