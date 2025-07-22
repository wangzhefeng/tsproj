# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_trend.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-01
# * Version     : 1.0.060118
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
plt.style.use("seaborn-v0_8-whitegrid")
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
scatter_plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Trues",
)
pred_line_plot_params = dict(
    linewidth=3,
    legend=True,
    label="Predict",
)
fore_line_plot_params = dict(
    color="C3",
    linewidth=3,
    legend=True,
    label="Forecast",
)
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


# ------------------------------
# data path
# ------------------------------
data_dir = Path("./dataset/ts_course_data")
logger.info(f"data_dir: {data_dir}")

# ------------------------------
# data
# ------------------------------
tunnel = pd.read_csv(
    data_dir / "tunnel.csv", 
    index_col="Day",
    parse_dates=["Day"]
)
tunnel = tunnel.to_period()
logger.info(f"tunnel: \n{tunnel}")

# ------------------------------
# moving average plots
# ------------------------------
moving_average = tunnel.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()
logger.info(f"moving_average: \n{moving_average}")

# meaning average plot
fig, ax = plt.subplots()
tunnel.plot(ax=ax, **scatter_plot_params)
moving_average.plot(ax=ax, **pred_line_plot_params)
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Tunnel Traffic - 365-Day Moving Average")
plt.show();


# ------------------------------
# deterministic process
# ------------------------------
dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,
    order=1,
    drop=True,
)
X = dp.in_sample()
y = tunnel["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_pred: \n{y_pred}")

# deterministic process plot
fig, ax = plt.subplots()
ax = y.plot(**scatter_plot_params)
ax = y_pred.plot(**pred_line_plot_params)
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Tunnel Traffic - Linear Trend")
plt.show();

# ------------------------------
# forecasting
# ------------------------------
X = dp.out_of_sample(steps=35)
logger.info(f"X: \n{X}")
y_fore = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_fore: \n{y_fore}")

fig, ax = plt.subplots()
ax = tunnel["2005-05":].plot(ax=ax, **scatter_plot_params)
ax = y_pred["2005-05":].plot(ax=ax, **pred_line_plot_params)
ax = y_fore.plot(ax=ax, **fore_line_plot_params)
_ = ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Tunnel Traffic - Linear Trend Forecast")
plt.show();



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
