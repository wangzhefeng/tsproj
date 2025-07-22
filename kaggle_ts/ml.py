# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_ml.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-02
# * Version     : 1.0.060218
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
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
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
    # label="Trues",
)
pred_line_plot_params = dict(
    linewidth=2,
    legend=True,
    label="Predict",
)
fore_line_plot_params = dict(
    color="C3",
    linewidth=2,
    legend=True,
    label="Forecast",
)
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    
    return ax


def make_lags(series, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': series.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1
    )


def make_multistep_target(series, steps):
    return pd.concat(
        {
            f'y_step_{i + 1}': series.shift(-i)
            for i in range(steps)
        },
        axis=1
    )


# ------------------------------
# data path
# ------------------------------
data_dir = Path("./dataset/ts_course_data")
logger.info(f"data_dir: {data_dir}")


# ------------------------------
# data
# ------------------------------
flu_trends = pd.read_csv(data_dir / "flu-trends.csv")
flu_trends.set_index(pd.PeriodIndex(flu_trends["Week"], freq="W"), inplace=True)
flu_trends.drop("Week", axis=1, inplace=True)
logger.info(f"flu_trends: \n{flu_trends} \nflu_trends.columns: \n{flu_trends.columns}")

# data preprocessing
y = flu_trends["FluVisits"].copy()
X = make_lags(y, lags=4).fillna(0.0)
logger.info(f"X: \n{X} \nX.tail(10): \n{X.tail(10)}")
logger.info(f"y: \n{y} \ny.tail(10): \n{y.tail(10)}")

# forecast(steps)
y = make_multistep_target(y, steps=8).dropna()
y, X = y.align(X, join="inner", axis=0)
logger.info(f"X: \n{X} \nX.columns: \n{X.columns}")
logger.info(f"y: \n{y} \ny.columns: \n{y.columns}")


# ------------------------------
# Multioutput model
# ------------------------------
logger.info(f"{'-' * 40}")
logger.info("Multioutput strategy")
logger.info(f"{'-' * 40}")
# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    shuffle=False, 
    random_state=42
)
logger.info(f"y_train \n{y_train}")
logger.info(f"y_test \n{y_test}")

# model fit/training
model = LinearRegression()
model.fit(X_train, y_train)
y_fit = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y.columns)
logger.info(f"y_fit: \n{y_fit}")
# model forecast
y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y.columns)
logger.info(f"y_pred: \n{y_pred}")
# model evaluation
train_rmse = root_mean_squared_error(y_train, y_fit)
test_rmse = root_mean_squared_error(y_test, y_pred)
logger.info(f"Train RMSE: {train_rmse:.2f}")
logger.info(f"Test RMSE: {test_rmse:.2f}")

# result plot
palette = dict(palette="husl", n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends["FluVisits"][y_fit.index].plot(**scatter_plot_params, ax=ax1)
ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(["FluVisits (train)", "Forecast"])
ax2 = flu_trends["FluVisits"][y_pred.index].plot(**scatter_plot_params, ax=ax2)
ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(["FluVisits (test)", "Forecast"])
plt.show();


# ------------------------------
# Direct strategy
# ------------------------------
logger.info(f"{'-' * 40}")
logger.info("Direct strategy")
logger.info(f"{'-' * 40}")
from sklearn.multioutput import MultiOutputRegressor

# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    shuffle=False, 
    random_state=42
)
logger.info(f"y_train \n{y_train}")
logger.info(f"y_test \n{y_test}")

# model fit/training
model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)
y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
logger.info(f"y_fit: \n{y_fit}")

# model forecast
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
logger.info(f"y_pred: \n{y_pred}")

# model evaluation
train_rmse = root_mean_squared_error(y_train, y_fit)
test_rmse = root_mean_squared_error(y_test, y_pred)
logger.info(f"Train RMSE: {train_rmse:.2f}")
logger.info(f"Test RMSE: {test_rmse:.2f}")

# result plot
palette = dict(palette="husl", n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends["FluVisits"][y_fit.index].plot(**scatter_plot_params, ax=ax1)
ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(["FluVisits (train)", "Forecast"])
ax2 = flu_trends["FluVisits"][y_pred.index].plot(**scatter_plot_params, ax=ax2)
ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(["FluVisits (test)", "Forecast"])
plt.show();


# ------------------------------
# Recursive strategy
# ------------------------------
# TODO


# ------------------------------
# Direct Recursive strategy
# ------------------------------
# TODO




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
