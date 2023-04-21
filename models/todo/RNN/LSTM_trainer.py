# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LSTM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032805
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import gc
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from RNNModel.LSTMRegressor import LSTMRegressor


# global variable
LOGGING_LABEL = __file__.split("/")[-1][:-3]
lookback = int(5 * (60 / 10))  # 5 hours lookback to prediction
batch_size = 32
num_epochs = 20
learning_rate = 1e-3


# ------------------------------
# data
# ------------------------------
# data download
data_url, data_path = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00616/Tetuan%20City%20power%20consumption.csv", 
    "./data/Tetuan City power consumption.csv",
)
if not os.path.exists(data_path):
    os.system(f"wget {data_url}")

# data
data = pd.read_csv(data_path)
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.set_index("DateTime")
data.columns = [col.strip() for col in data.columns]

# data preprocessing
features_cols = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
target_col = "Zone 1 Power Consumption"
X = data[features_cols].values
Y = data[target_col].values
num_features = X.shape[1]

X_org, Y_org = [], []
for i in range(0, X.shape[0] - lookback, 1):
    X_org.append(X[i:i + lookback])
    Y_org.append(Y[i + lookback])
X_org = torch.tensor(np.array(X_org), dtype = torch.float32)
Y_org = torch.tensor(np.array(Y_org), dtype = torch.float32)

# data split
X_train, Y_train = X_org[:50000], Y_org[:50000]
X_test, Y_test = X_org[50000:], Y_org[50000:]

# target scale
mean, std = Y_train.mean(), Y_train.std()
Y_train_scaled, Y_test_scaled = (Y_train - mean) / std, (Y_test - mean) / std

# dataset
train_dataset = TensorDataset(X_train, Y_train_scaled)
test_dataset = TensorDataset(X_test, Y_test_scaled)

# dataloader
train_loader = DataLoader(
    train_dataset, 
    batch_size = batch_size,
    shuffle = False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)

# 内存回收
del X, Y
gc.collect()

# ------------------------------
# model
# ------------------------------
# model
lstm_reg = LSTMRegressor(num_features = num_features)
print(lstm_reg)
for layer in lstm_reg.children():
    print(f"Layer: {layer}")
    print("Parameters:")
    for param in layer.parameters():
        print("\t", param.shape)
    print()

# loss
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(lstm_reg.parameters(), lr = learning_rate)

# ------------------------------
# model training
# ------------------------------
for epoch in range(num_epochs):
    train_losses = []
    for X, Y in tqdm(train_loader):
        # forward
        Y_outputs = lstm_reg(X)
        loss = loss_fn(Y_outputs.ravel(), Y)
        train_losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train Loss: {torch.tensor(train_losses).mean()}")
    # validation
    with torch.no_grad():
        val_losses = []
        for X, Y in test_loader:
            preds = lstm_reg(X)
            val_loss = loss_fn(preds.ravel(), Y)
            val_losses.append(val_loss)
        print(f"Valid Loss: {torch.tensor(val_losses).mean()}")

# ------------------------------
# model testing
# ------------------------------
test_preds = lstm_reg(X_test)
test_preds = (test_preds * std) + mean
print("Test MSE : {:.2f}".format(mean_squared_error(test_preds.detach().numpy().squeeze(), Y_test.detach().numpy())))
print("Test R2 Score : {:.2f}".format(r2_score(test_preds.detach().numpy().squeeze(), Y_test.detach().numpy())))

# ------------------------------
# model testing view
# ------------------------------
data_df_final = data[50000:].copy()
data_df_final["Zone 1 Power Consumption Prediction"] = [None] * lookback + test_preds.detach().numpy().squeeze().tolist()
data_df_final.plot(
    y = ["Zone 1 Power Consumption", "Zone 1 Power Consumption Prediction"],
    figsize = (18, 7)
)
plt.grid(which = 'minor', linestyle = ':', linewidth = '0.5', color = 'black');
plt.show()




# 测试代码 main 函数
def main():
    """
    # data
    print(f"Data Columns: {data.columns.values.tolist()}")
    print(f"Data Shape: {data.shape}")
    print(data.head())
    # data view of 2017-12 data and 2017-12-01 data
    data.loc["2017-12"].plot(
        y = "Zone 1 Power Consumption", 
        figsize = (18, 7), 
        color = "blue", 
        grid = True
    )
    plt.grid(which = "minor", linestyle = ":", linewidth = "0.5", color = "black")
    data.loc["2017-12-1"].plot(
        y = "Zone 1 Power Consumption", 
        figsize = (18, 7), 
        color = "blue", 
        grid = True
    )
    plt.grid(which = "minor", linestyle = ":", linewidth = "0.5", color = "black");
    plt.show()
    """
    pass

if __name__ == "__main__":
    main()
