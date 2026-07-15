# -*- coding: utf-8 -*-
# Maximillian Ermler
# IOE 562 Project
# Deep Learning — LSTM for Toledo Edison Demand Forecasting

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# LOAD DATA
# ============================================================
TOL = pd.read_csv("/Users/maxermler/Documents/University of Michigan (2025-2026)/IOE562/Project/project files/TOLEDO data/toledo_clean.csv")
TOL['datetime'] = pd.to_datetime(TOL['datetime'])
TOL = TOL.sort_values('datetime').reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# lagged demand
TOL['lag_day_demand']  = TOL['demand'].shift(24)
TOL['lag_week_demand'] = TOL['demand'].shift(168)

# sine/cosine
TOL['hour_sin']  = np.sin(2 * np.pi * TOL['datetime'].dt.hour / 24)
TOL['hour_cos']  = np.cos(2 * np.pi * TOL['datetime'].dt.hour / 24)
TOL['week_sin']  = np.sin(2 * np.pi * TOL['datetime'].dt.dayofweek / 7)
TOL['week_cos']  = np.cos(2 * np.pi * TOL['datetime'].dt.dayofweek / 7)
TOL['month_sin'] = np.sin(2 * np.pi * TOL['datetime'].dt.month / 12)
TOL['month_cos'] = np.cos(2 * np.pi * TOL['datetime'].dt.month / 12)

# boolean indicators
TOL['is_weekend'] = (TOL['datetime'].dt.dayofweek >= 5).astype(int)

# holidays
holidays = pd.to_datetime([
    "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",
    "2019-05-27", "2020-05-25", "2021-05-31", "2022-05-30", "2023-05-29", "2024-05-27", "2025-05-26",
    "2019-07-04", "2020-07-04", "2021-07-04", "2022-07-04", "2023-07-04", "2024-07-04", "2025-07-04",
    "2019-09-02", "2020-09-07", "2021-09-06", "2022-09-05", "2023-09-04", "2024-09-02", "2025-09-01",
    "2019-11-28", "2020-11-26", "2021-11-25", "2022-11-24", "2023-11-23", "2024-11-28", "2025-11-27",
    "2019-12-25", "2020-12-25", "2021-12-25", "2022-12-25", "2023-12-25", "2024-12-25", "2025-12-25"
])
TOL['is_holiday'] = TOL['datetime'].dt.date.astype('datetime64[ns]').isin(holidays).astype(int)

# COVID indicator
TOL['is_covid'] = ((TOL['datetime'] >= '2020-03-01') & 
                   (TOL['datetime'] <= '2021-06-01')).astype(int)

# heatwave 
temp_90 = TOL['temperature'].quantile(0.9)
TOL['is_heatwave'] = ((TOL['temperature'] > temp_90) & 
                      (TOL['temperature'].shift(24) > temp_90)).astype(int)

# temperature
TOL['temp_sq'] = TOL['temperature'] ** 2
TOL['temp_cb'] = TOL['temperature'] ** 3

# drop NAs from lags
TOL = TOL.dropna().reset_index(drop=True)

# ============================================================
# FEATURE COLUMNS 
# ============================================================
feature_cols = [
    'temperature', 'humidity', 'apparent_temp', 'precipitation',
    'rain', 'snowfall', 'snow_depth', 'cloud_cover', 'et0',
    'vpd', 'wind_speed', 'wind_direction', 'wind_gusts',
    'soil_temp', 'shortwave_rad', 'direct_rad', 'is_day',
    'soil_moisture', 'direct_rad_inst',
    'lag_day_demand', 'lag_week_demand',
    'hour_sin', 'hour_cos', 'week_sin', 'week_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_holiday', 'is_covid', 'is_heatwave',
    'temp_sq', 'temp_cb'
]

# ============================================================
# TRAIN / VAL / TEST SPLIT , same as R
# ============================================================
train = TOL[TOL['datetime'] < '2023-01-01'].copy()
val   = TOL[(TOL['datetime'] >= '2023-01-01') & (TOL['datetime'] < '2023-11-01')].copy()
test  = TOL[TOL['datetime'] >= '2023-11-01'].copy()

print(f"Train: {len(train)} rows")
print(f"Val:   {len(val)} rows")
print(f"Test:  {len(test)} rows")

# ============================================================
# SCALE fit on train only
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train[feature_cols])
X_val   = scaler_X.transform(val[feature_cols])
X_test  = scaler_X.transform(test[feature_cols])

y_train = scaler_y.fit_transform(np.log(train[['demand']]))
y_val   = scaler_y.transform(np.log(val[['demand']]))
y_test  = scaler_y.transform(np.log(test[['demand']]))

# ============================================================
# CREATE SEQUENCES 
# ============================================================
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

seq_length = 24

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   seq_length)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_length)

print(f"Train sequences: {X_train_seq.shape}")
print(f"Val sequences:   {X_val_seq.shape}")
print(f"Test sequences:  {X_test_seq.shape}")

# convert to tensors
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_seq,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val_seq,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=256, shuffle=False)

# ============================================================
# LSTM MODEL 
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn

# ============================================================
# INITIALIZE
# ============================================================
model     = LSTMModel(input_dim=len(feature_cols), hidden_dim=200,
                      layer_dim=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ============================================================
# TRAIN WITH EARLY STOPPING
# ============================================================
num_epochs        = 50
best_val_loss     = float('inf')
best_epoch        = 0
patience          = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        h0 = torch.zeros(2, xb.size(0), 200)
        c0 = torch.zeros(2, xb.size(0), 200)
        outputs, h0, c0 = model(xb, h0, c0)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        h0_val = torch.zeros(2, X_val_t.size(0), 200)
        c0_val = torch.zeros(2, X_val_t.size(0), 200)
        val_out, _, _ = model(X_val_t, h0_val, c0_val)
        val_loss = criterion(val_out, y_val_t).item()

    print(f"Epoch [{epoch+1}/{num_epochs}]  "
          f"Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        best_epoch        = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_lstm.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}, best was epoch {best_epoch}")
            break

model.load_state_dict(torch.load('best_lstm.pt'))
print(f"\nBest model from epoch {best_epoch} with val loss {best_val_loss:.4f}")

# ============================================================
# EVALUATE
# ============================================================
model.eval()
with torch.no_grad():
    h0_test = torch.zeros(2, X_test_t.size(0), 200)
    c0_test = torch.zeros(2, X_test_t.size(0), 200)
    pred_scaled, _, _ = model(X_test_t, h0_test, c0_test)

pred_log    = scaler_y.inverse_transform(pred_scaled.numpy())
pred_demand = np.exp(pred_log)
actual      = np.exp(scaler_y.inverse_transform(y_test_seq))

RMSE_lstm = np.sqrt(np.mean((actual - pred_demand)**2))
MAE_lstm  = np.mean(np.abs(actual - pred_demand))

print(f"LSTM Test RMSE: {RMSE_lstm:.1f} KW")
print(f"LSTM Test MAE:  {MAE_lstm:.1f} KW")

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(actual[:500],      label='Actual Demand')
plt.plot(pred_demand[:500], label='LSTM Predicted', linestyle='--')
plt.title('LSTM Predictions vs Actual Demand (first 500 test hours)')
plt.xlabel('Hour')
plt.ylabel('Demand (KW)')
plt.legend()
plt.show()