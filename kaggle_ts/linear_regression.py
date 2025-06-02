# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_linear_regression.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-01
# * Version     : 1.0.060118
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from pathlib import Path
from warnings import simplefilter
simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
plt.style.use("seaborn-v0_8-whitegrid")  # print(plt.style.available)
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4.5),
    titleweight="bold",
    titlesize=18,
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
from sklearn.linear_model import LinearRegression
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


# data path
data_dir = Path("./dataset/ts_course_data")
logger.info(f"data_dir: {data_dir}")

# ------------------------------
# linear regression: book sales
# ------------------------------
# data
df = pd.read_csv(
    data_dir / "book_sales.csv", 
    index_col="Date", 
    parse_dates=["Date"]
)
df = df.drop("Paperback", axis=1)
logger.info(f"df: \n{df.head()}")

# time-step features
df["Time"] = np.arange(len(df.index))
logger.info(f"df: \n{df.head()}")
# time-step feature plot
# fig, ax = plt.subplots()
# ax.plot("Time", "Hardcover", data=df, color="0.75")
# ax = sns.regplot(x="Time", y="Hardcover", data=df, ci=None, scatter_kws=dict(color="0.25"))
# ax.set_xlabel("Time")
# ax.set_ylabel("Hardcover")
# ax.set_title("Time Plot of Hardcover Sales")
# plt.show();

# lag features
df["Lag_1"] = df["Hardcover"].shift(1)
df = df.reindex(columns=["Hardcover", "Time", "Lag_1"])
logger.info(f"df: \n{df.head()}")
# lag feature plot
# fig, ax = plt.subplots()
# ax = sns.regplot(x="Lag_1", y="Hardcover", data=df, ci=None, scatter_kws=dict(color="0.25"))
# ax.set_aspect("equal")
# ax.set_xlabel("Lag_1")
# ax.set_ylabel("Hardcover")
# ax.set_title("Lag Plot of Hardcover Sales")
# plt.show();




# ------------------------------
# linear regression: tunnel traffic 
# ------------------------------
# data
tunnel = pd.read_csv(
    data_dir / "tunnel.csv", 
    index_col="Day",
    parse_dates=["Day"]
)
tunnel = tunnel.to_period()
logger.info(f"tunnel: \n{tunnel}")


# time-step features: linear model
# ------------------------------
df = tunnel.copy()
df["Time"] = np.arange(len(tunnel.index))
logger.info(f"df: \n{df}")

X = df.loc[:, ["Time"]]
y = df.loc[:, "NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_pred: \n{y_pred}")

# time-step feature model plot
fig, ax = plt.subplots()
ax.plot(X["Time"], y, ".", color="0.25")
ax.plot(X["Time"], y_pred)
# ax.set_aspect("equal")
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Lag Plot of Tunnel Traffic")
plt.show();

fig, ax = plt.subplots()
ax = y.plot(**plot_params)
ax = y_pred.plot()
plt.show();


# lag features: linear model
# ------------------------------
df["Lag_1"] = df["NumVehicles"].shift(1)
df = df.dropna(axis=0)
logger.info(f"df: \n{df}")

X = df[["Lag_1"]]
y = df["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_pred: \n{y_pred}")

# lag feature model plot
fig, ax = plt.subplots()
ax.plot(X["Lag_1"], y, ".", color="0.25")
ax.plot(X["Lag_1"], y_pred)
ax.set_aspect("equal")
ax.set_xlabel("Lag_1")
ax.set_ylabel("NumVehicles")
ax.set_title("Lag Plot of Tunnel Traffic")
plt.show();


fig, ax = plt.subplots()
ax = y.plot(**plot_params)
ax = y_pred.plot()
plt.show();

# time-step and lag features: linear model
# ------------------------------
df["Lag_1"] = df["NumVehicles"].shift(1)
df = df.dropna(axis=0)
logger.info(f"df: \n{df}")

X = df[["Time", "Lag_1"]]
y = df["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_pred: \n{y_pred}")

# lag feature model plot
fig, ax = plt.subplots()
ax = y.plot(**plot_params)
ax = y_pred.plot()
plt.show();







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
