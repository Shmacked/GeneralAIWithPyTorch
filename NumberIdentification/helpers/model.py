import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns # Optional: uv add seaborn for prettier colors

def train_model(model, train_loader, device, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_accuracy = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        epoch_accuracy = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            batch_samples = target.size(0)
            batch_loss = loss.item()
            batch_correct_samples = (predicted == target).sum().item()
            batch_accuracy = batch_correct_samples / batch_samples

            epoch_loss += batch_loss
            epoch_correct += batch_correct_samples
            epoch_samples += batch_samples
        
        epoch_accuracy = epoch_correct / epoch_samples

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

        print(f"Epoch {epoch} | Loss {epoch_loss:.4f} | Average Loss {total_loss / (epoch + 1):.4f} | Accuracy {epoch_accuracy:.4f}")
    end_time = time.time()
    total_accuracy = total_correct / total_samples
    print(f"Total Loss {total_loss:.4f} | Total Samples {total_samples:.4f} | Total Correct Samples {total_correct:.4f} | Total Accuracy {total_accuracy:.4f}")
    return model, total_accuracy, total_loss, end_time - start_time


def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_accuracy += (total_correct / total_samples)
    total_accuracy = total_correct / total_samples
    print(f"Total Loss {total_loss:.4f} | Total Samples {total_samples:.4f} | Total Correct Samples {total_correct:.4f} | Total Accuracy {total_accuracy:.4f}")
    return model, total_accuracy, total_loss

def plot_confusion_matrix(model, test_loader, device):
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Where is the model confused?')
    plt.show()

def predict(model, data, device):
    model.eval()
    with torch.no_grad():
        output = model(data)
        return output
    return None

def show_image(image):
    image = image.squeeze(0)
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * 255.0
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.show()
