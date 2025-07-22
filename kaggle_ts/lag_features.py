# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_lag_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-02
# * Version     : 1.0.060201
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
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
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


def __lag_plot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = __lag_plot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    
    return fig


def make_lags(series, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': series.shift(i)
            for i in range(lead_time, lags + lead_time)
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
flu_trends = pd.read_csv(
    data_dir / "flu-trends.csv", 
    # parse_dates={"Date": ["Year", "Month", "Day"]},
    # index_col="Date",
)
flu_trends.set_index(pd.PeriodIndex(flu_trends["Week"], freq="W"), inplace=True)
flu_trends.drop("Week", axis=1, inplace=True)
logger.info(f"flu_trends: \n{flu_trends} \nflu_trends.columns: \n{flu_trends.columns}")

# fig, ax = plt.subplots()
# ax = flu_trends["FluVisits"].plot(ax=ax, **scatter_plot_params)
# ax.set_xlabel("Week")
# ax.set_ylabel("Office Visits")
# ax.set_title("Flu Trends")
# plt.show();

# lag plots
# _ = plot_lags(flu_trends["FluVisits"], lags=12, nrows = 2)
# plt.show();

# pacf plot
# _ = plot_pacf(flu_trends['FluVisits'], lags=12)
# plt.show();


# training data
X = make_lags(flu_trends["FluVisits"], lags=4)
X = X.fillna(0.0)
y = flu_trends["FluVisits"].copy()
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False, random_state=42)

# model fit/training
model = LinearRegression()
model.fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=y_train.index)

# model forecast
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# result plot
# fig, ax = plt.subplots()
# ax = y_train.plot(ax=ax, label="train", **scatter_plot_params)
# ax = y_test.plot(ax=ax, label="test", **scatter_plot_params)
# ax = y_fit.plot(ax=ax, **pred_line_plot_params)
# ax = y_pred.plot(ax=ax, **fore_line_plot_params)
# _ = ax.legend()
# ax.set_xlabel("Time")
# ax.set_ylabel("NumVehicles")
# ax.set_title("Flu Visits Lag Forecast")
# plt.show();

# ------------------------------
# leading indicators
# ------------------------------
# fig, ax = plt.subplots()
# ax = flu_trends.plot(ax = ax, y = ["FluCough", "FluVisits"], secondary_y="FluCough")
# ax.set_xlabel("Week")
# ax.set_ylabel("Value")
# ax.set_title("Flu Visits and Flu Cough")
# plt.show();

search_terms = [
    "FluContagious", "FluCough", "FluFever", "InfluenzaA", 
    "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"
]
X0 = make_lags(flu_trends[search_terms], lags=3)
X0.columns = [" ".join(col).strip() for col in X0.columns.values]
logger.info(f"X0: \n{X0} \nX0.columns: \n{X0.columns}")
X1 = make_lags(flu_trends["FluVisits"], lags=4)
logger.info(f"X1: \n{X1} \nX0.columns: \n{X1.columns}")
X = pd.concat([X0, X1], axis=1).fillna(0.0)
y = flu_trends["FluVisits"].copy()
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False, random_state=42)

# model fit/training
model = LinearRegression()
model.fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=y_train.index)

# model forecast
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# result plot
fig, ax = plt.subplots()
# ax = y_train.plot(ax=ax, label="train", **scatter_plot_params)
ax = y_test.plot(ax=ax, label="test", **scatter_plot_params)
# ax = y_fit.plot(ax=ax, **pred_line_plot_params)
ax = y_pred.plot(ax=ax, **fore_line_plot_params)
_ = ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("NumVehicles")
ax.set_title("Flu Visits Lag Forecast")
plt.show();





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
