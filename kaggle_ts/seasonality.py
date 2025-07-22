# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_seasonality.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-01
# * Version     : 1.0.060121
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
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import (
    CalendarFourier, 
    DeterministicProcess
)
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


# TODO
def seasonal_indicators():
    pass


def fourier_features(series, freq: str="A", order: int=10, steps: int=10):
    # 10 sin/cos pairs for "A"nnual seasonality
    fourier = CalendarFourier(freq=freq, order=order)
    # deterministic process
    dp = DeterministicProcess(
        index=series.index,
        constant=True,               # dummy feature for bias (y-intercept)
        order=1,                     # trend (order 1 means linear)
        seasonal=True,               # weekly seasonality (indicators)
        additional_terms=[fourier],  # annual seasonality (fourier)
        drop=True,                   # drop terms to avoid collinearity
    )
    
    return dp


def seasonal_plot(X, y, period, freq, ax=None):
    """
    季节性图
    # annotations: https://stackoverflow.com/a/49238256/5769929
    """
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
        
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    """
    周期图

    Args:
        ts (_type_): _description_
        detrend (str, optional): _description_. Defaults to 'linear'.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    
    return ax


# def fourier_features(index, freq, order):
#     time = np.arange(len(index), dtype=np.float32)
#     k = 2 * np.pi * (1 / freq) * time
#     features = {}
#     for i in range(1, order + 1):
#         features.update({
#             f"sin_{freq}_{i}": np.sin(i * k),
#             f"cos_{freq}_{i}": np.cos(i * k),
#         })
    
#     return pd.DataFrame(features, index=index)


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

X = tunnel.copy()
# days within a week
X["day"] = X.index.dayofweek  # the x-axis(freq)
X["week"] = X.index.week  # the seasonal period(period)
# days within a year
X["dayofyear"] = X.index.dayofyear  # the x-axis (freq)
X["year"] = X.index.year  # the seasonal period(period)

# ------------------------------
# seasonal plot
# ------------------------------
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
seasonal_plot(X, y="NumVehicles", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear", ax=ax1)
plt.show();

plot_periodogram(tunnel["NumVehicles"])
plt.show();


# ------------------------------
# model
# ------------------------------
dp = fourier_features(tunnel, freq="A", order=10)
# create features for dates in tunnel.index
X = dp.in_sample()
y = tunnel["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# model training
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=y.index)

# model forecasting
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

fig, ax = plt.subplots()
ax = y.plot(ax=ax, **scatter_plot_params)
ax = y_pred.plot(ax=ax, **pred_line_plot_params)
ax = y_fore.plot(ax=ax, **fore_line_plot_params)
_ = ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Tunnel Traffic - Seasonal Forecast")
plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
