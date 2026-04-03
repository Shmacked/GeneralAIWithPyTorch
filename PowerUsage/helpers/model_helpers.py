import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



def make_lag_df(df, features, window):
    df_copy = df.copy()
    for feature in features:
        for i in range(window, 0, -1):
            col = f"{feature}_t_{i}"
            df_copy[col] = df_copy[feature].shift(i)
    df_copy = df_copy.drop(features, axis=1)
    df_copy = df_copy.dropna()
    return df_copy


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
        print(f"{model_name} - loss for epoch {epoch + 1}: {epoch_loss}")
    print()
    return model


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
