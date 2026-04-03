import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from helpers.model_helpers import make_lag_df, df_to_loader, train_model, predict, model_size_bytes
from models import RNN, GRU, LSTM

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv("datasets/household_power_train.csv", index_col="Timestamp", parse_dates=["Timestamp"])
print(df)


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


lagged_df =  make_lag_df(df, original_features, window)
lagged_features = lagged_df.columns

lagged_df["Global_active_power"] = df.loc[lagged_df.index, "Global_active_power"].astype("float32")

train_loader = df_to_loader(lagged_df, 32, window, lagged_features)

rnn = RNN().to(device)
gru = GRU().to(device)
lstm = LSTM().to(device)

mse = nn.MSELoss()
optimizer_rnn = optim.Adam(rnn.parameters(), lr=0.001)
optimizer_gru = optim.Adam(gru.parameters(), lr=0.001)
optimizer_lstm = optim.Adam(lstm.parameters(), lr=0.001)

rnn = train_model(rnn, mse, optimizer_rnn, train_loader, 10, "RNN", device)
gru = train_model(gru, mse, optimizer_rnn, train_loader, 10, "GRU", device)
lstm = train_model(lstm, mse, optimizer_rnn, train_loader, 10, "LSTM", device)

df_test = pd.read_csv("datasets/household_power_test.csv", index_col="Timestamp", parse_dates=["Timestamp"])
df_test_lagged = make_lag_df(df_test, original_features, window)
lagged_features = df_test_lagged.columns
df_test_lagged["Global_active_power"] = df_test.loc[df_test_lagged.index, "Global_active_power"].astype("float32")
test_loader = df_to_loader(df_test_lagged, 32, window, lagged_features, shuffle=False)

rnn_preds, rnn_targs = predict(rnn, test_loader, device)
gru_preds, gru_targs = predict(gru, test_loader, device)
lstm_preds, lstm_targs = predict(lstm, test_loader, device)

rnn_mse = mean_squared_error(rnn_preds, rnn_targs)
gru_mse = mean_squared_error(gru_preds, gru_targs)
lstm_mse = mean_squared_error(lstm_preds, lstm_targs)
print(f"{rnn_mse} - {gru_mse} - {lstm_mse}")

print(rnn_preds[-2], rnn_targs[-2])
print(gru_preds[-2], gru_targs[-2])
print(lstm_preds[-2], lstm_targs[-2])

print(model_size_bytes(rnn))
print(model_size_bytes(gru))
print(model_size_bytes(lstm))
