import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from helpers.model_helpers import make_lag_df, df_to_loader, train_model, predict, model_size_bytes, plot_loss, plot_regression_margin
from models import RNN, GRU, LSTM

from pathlib import Path

if not Path("models").exists():
    Path("models").mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv("data/household_power_train.csv", index_col="Timestamp", parse_dates=["Timestamp"])

original_features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

window = 24

# 1. Get your features (the lags) from the function we optimized
lagged_df = make_lag_df(df, original_features, window)
lagged_features = lagged_df.columns
# 2. Get the target values (Global_active_power) as a separate Series
target_series = df.loc[lagged_df.index, "Global_active_power"].astype("float32")
# 3. COMBINE THEM ALL AT ONCE (This prevents fragmentation)
lagged_df = pd.concat([lagged_df, target_series], axis=1)
# 4. Final 'De-frag' (Optional but recommended)
lagged_df = lagged_df.copy()

train_loader = df_to_loader(lagged_df, 32, window, lagged_features, shuffle=True)

rnn = RNN().to(device)
gru = GRU().to(device)
lstm = LSTM().to(device)

mse = nn.MSELoss()

rnn_loss_history = []
gru_loss_history = []
lstm_loss_history = []

if not Path("models/rnn.pth").exists():
    optimizer_rnn = optim.Adam(rnn.parameters(), lr=0.001)
    rnn, rnn_loss_history = train_model(rnn, mse, optimizer_rnn, train_loader, 10, "RNN", device)
    torch.save(rnn.state_dict(), "models/rnn.pth")
else:
    rnn.load_state_dict(torch.load("models/rnn.pth"))
if not Path("models/gru.pth").exists():
    optimizer_gru = optim.Adam(gru.parameters(), lr=0.001)
    gru, gru_loss_history = train_model(gru, mse, optimizer_gru, train_loader, 10, "GRU", device)
    torch.save(gru.state_dict(), "models/gru.pth")
else:
    gru.load_state_dict(torch.load("models/gru.pth"))
if not Path("models/lstm.pth").exists():
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.001)
    lstm, lstm_loss_history = train_model(lstm, mse, optimizer_lstm, train_loader, 10, "LSTM", device)
    torch.save(lstm.state_dict(), "models/lstm.pth")
else:
    lstm.load_state_dict(torch.load("models/lstm.pth"))

df_test = pd.read_csv("data/household_power_test.csv", index_col="Timestamp", parse_dates=["Timestamp"])
# 1. Get your features (the lags) from the function we optimized
df_test_lagged = make_lag_df(df_test, original_features, window)
df_test_features = df_test_lagged.columns
# 2. Get the target values (Global_active_power) as a separate Series
target_series = df_test.loc[df_test_lagged.index, "Global_active_power"].astype("float32")
# 3. COMBINE THEM ALL AT ONCE (This prevents fragmentation)
df_test_lagged = pd.concat([df_test_lagged, target_series], axis=1)
# 4. Final 'De-frag' (Optional but recommended)
df_test_lagged = df_test_lagged.copy()
test_loader = df_to_loader(df_test_lagged, 32, window, df_test_features, shuffle=False)

rnn_preds, rnn_targs = predict(rnn, test_loader, device)
gru_preds, gru_targs = predict(gru, test_loader, device)
lstm_preds, lstm_targs = predict(lstm, test_loader, device)

print()

print("2nd to last prediction and target:")
print(f"RNN: {rnn_preds[-2]} - {rnn_targs[-2]}")
print(f"GRU: {gru_preds[-2]} - {gru_targs[-2]}")
print(f"LSTM: {lstm_preds[-2]} - {lstm_targs[-2]}")

print()

print(f"RNN size: {model_size_bytes(rnn) / 1024**2:.2f} MB")
print(f"GRU size: {model_size_bytes(gru) / 1024**2:.2f} MB")
print(f"LSTM size: {model_size_bytes(lstm) / 1024**2:.2f} MB")

if len(rnn_loss_history) > 0 and len(gru_loss_history) > 0 and len(lstm_loss_history) > 0:
    plot_loss(rnn_loss_history, gru_loss_history, lstm_loss_history)

show_plots = False
if show_plots:
    plot_regression_margin(rnn_targs, rnn_preds, "RNN", margin=0.2)
    plot_regression_margin(gru_targs, gru_preds, "GRU", margin=0.2)
    plot_regression_margin(lstm_targs, lstm_preds, "LSTM", margin=0.2)

print()

rnn_mse = mean_squared_error(rnn_preds, rnn_targs)
gru_mse = mean_squared_error(gru_preds, gru_targs)
lstm_mse = mean_squared_error(lstm_preds, lstm_targs)
print(f"MSE: RNN: {rnn_mse} - GRU: {gru_mse} - LSTM: {lstm_mse}")

rnn_mae = mean_absolute_error(rnn_targs, rnn_preds)
gru_mae = mean_absolute_error(gru_targs, gru_preds)
lstm_mae = mean_absolute_error(lstm_targs, lstm_preds)
print(f"MAE: RNN: {rnn_mae} - GRU: {gru_mae} - LSTM: {lstm_mae}")