import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as M
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from helpers.model_size import profile_model_size
from helpers.model import train_model, evaluate_model, predict, show_image
from simple_mlp import SimpleMLP
from simple_resnet import SimpleResNet

import random

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

test_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_transform = transforms.Compose([
    transforms.RandomRotation(15), # Learn to handle tilts
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Learn to handle off-center
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model1 = SimpleMLP().to(device)
model2 = SimpleResNet().to(device)

# Profile the model size using an input size with the format: (batch_size, channels, width, height)
profile_size = (32, 1, 28, 28)
model_size1 = profile_model_size(model1, profile_size, device)
print(f"Model size: {model_size1 / 1024**2:.2f} MB")
model_size2 = profile_model_size(model2, profile_size, device)
print(f"Model size: {model_size2 / 1024**2:.2f} MB")

model1, accuracy1, loss1, time1 = train_model(model1, train_loader, device, 3)
model2, accuracy2, loss2, time2 = train_model(model2, train_loader, device, 3)
print(f"Model 1 Training Time: {time1:.2f} seconds")
print(f"Model 2 Training Time: {time2:.2f} seconds")

model1, accuracy1, loss1 = evaluate_model(model1, test_loader, device)
model2, accuracy2, loss2 = evaluate_model(model2, test_loader, device)

torch.save(model1.state_dict(), "model1.pth")
torch.save(model2.state_dict(), "model2.pth")

data = test_dataset[random.randint(0, len(test_dataset))][0].unsqueeze(0).to(device)
output1 = predict(model1, data, device)
output2 = predict(model2, data, device)

print(f"Model 1 Output: {output1.argmax(dim=1).item()}")
print(f"Model 2 Output: {output2.argmax(dim=1).item()}")

show_image(data)

if __name__ == "__main__":
    clean_up([
        model1,
        model2,
        data,
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
        train_transform,
        test_transform,
    ], globals())
