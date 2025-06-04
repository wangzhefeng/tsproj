# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ts_plot.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-03
# * Version     : 1.0.060315
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
train_scatter_plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Train trues",
)
test_scatter_plot_params = dict(
    color="C2",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Test trues",
)
fit_line_plot_params = dict(
    color="C0",
    linewidth=2,
    legend=True,
    label="Train preds",
)
pred_line_plot_params = dict(
    color="C1",
    linewidth=2,
    legend=True,
    label="Test preds",
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


def scatter_reg_plot(data, x, y, aspect: bool=False):
    """
    散点图回归拟合曲线
    """
    fig, ax = plt.subplots()
    # ax.plot(x, y, data=data, color="0.75")
    ax = sns.regplot(x=x, y=y, data=data, ci=None, scatter_kws=dict(color="0.25", s=9))
    if aspect:
        ax.set_aspect("equal")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x.capitalize()} Plot of {y.capitalize()}")
    plt.show();


def model_result_plot(y_train, y_test, y_fit, y_pred, y_fore, xlabel, ylabel, title=""):
    # result plot
    fig, ax = plt.subplots()
    if y_train is not None:
        ax = y_train.plot(ax=ax, **train_scatter_plot_params)
    if y_test is not None:
        ax = y_test.plot(ax=ax, **test_scatter_plot_params)
    if y_fit is not None:
        ax = y_fit.plot(ax=ax, **fit_line_plot_params)
    if y_pred is not None:
        ax = y_pred.plot(ax=ax, **pred_line_plot_params)
    if y_fore is not None:
        ax = y_fore.plot(ax=ax, **fore_line_plot_params)
    _ = ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
