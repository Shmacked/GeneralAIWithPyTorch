import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as M
from helpers.model_size import profile_model_size
from sklearn.metrics import accuracy_score, classification_report
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

model = SimpleMLP().to(device)

# Profile the model size using an input size with the format: (batch_size, channels, width, height)
profile_size = (32, 1, 28, 28)

model_size = profile_model_size(model, profile_size)
print(f"Model size: {model_size / 1024**2:.2f} MB")

model_resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model_resnet18.maxpool = nn.Identity() # removes maxpool layer