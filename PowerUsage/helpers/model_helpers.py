import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def make_lag_df(df, features, window):
    # 1. Start a list with the original index to keep things aligned
    new_cols = []
    
    for feature in features:
        for i in range(window, 0, -1):
            col_name = f"{feature}_t_{i}"
            # 2. Create the shifted series and give it the correct name
            shifted_series = df[feature].shift(i).rename(col_name)
            new_cols.append(shifted_series)
    
    # 3. Concatenate everything at once (extremely fast)
    # axis=1 means join side-by-side as columns
    lagged_df = pd.concat(new_cols, axis=1)
    
    # 4. Drop the rows with NaNs (the 'warm-up' period for the window)
    lagged_df = lagged_df.dropna()
    
    return lagged_df


def df_to_loader(df, batch, window, lagged_features, shuffle=True):
    N = df.shape[0]
    T = window
    F = int(len(lagged_features) / window)
    data = np.array(df[lagged_features])
    data = torch.tensor(data, dtype=torch.float32).view(N, T, F)
    target = torch.tensor(df["Global_active_power"].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    return loader


def train_model(model, loss_fn, optimizer, loader, epochs, model_name, device):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            predictions = model(data)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss)
        print(f"{model_name} - loss for epoch {epoch + 1}: {epoch_loss}")
    print()
    return model, loss_history


def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for batch, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            preds = model(data)
            all_preds.extend(preds.cpu().detach().numpy().ravel())
            all_targs.extend(targets.cpu().detach().numpy().ravel())
    return all_preds, all_targs

def model_size_bytes(model):
    size = 0
    for t in list(model.parameters()) + list(model.buffers()):
        size += t.numel() * t.element_size()
    return size

def plot_loss(rnn_losses, gru_losses, lstm_losses):
    # After your training loop finishes:
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label='RNN')
    plt.plot(gru_losses, label='GRU')
    plt.plot(lstm_losses, label='LSTM')

    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    # CRITICAL: This line opens the window!
    plt.show()

def plot_regression_margin(y_true, y_pred, model_name, margin=0.1):
    # 1. Convert to numpy arrays if they aren't already
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 2. Calculate the absolute error
    errors = np.abs(y_true - y_pred)
    
    # 3. Create a mask for points within the margin
    within_margin = errors <= margin
    
    # 4. Plotting
    plt.figure(figsize=(8, 8))
    
    # Plot points within margin (Green) and outside (Red)
    plt.scatter(y_true[within_margin], y_pred[within_margin], 
                color='green', alpha=0.5, label=f'Within {margin}kW')
    plt.scatter(y_true[~within_margin], y_pred[~within_margin], 
                color='red', alpha=0.3, label='Outside Margin')
    
    # Draw the "Perfect Prediction" diagonal line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
    
    # Draw the Margin Bounds (dashed lines)
    plt.plot([min_val, max_val], [min_val + margin, max_val + margin], 'g--', alpha=0.3)
    plt.plot([min_val, max_val], [min_val - margin, max_val - margin], 'g--', alpha=0.3)

    plt.xlabel('Actual Power Usage (kW)')
    plt.ylabel('Predicted Power Usage (kW)')
    plt.title(f'{model_name}: Accuracy Margin ({margin} kW)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
