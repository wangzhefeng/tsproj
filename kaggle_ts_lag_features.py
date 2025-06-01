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
reserve = pd.read_csv(
    data_dir / "reserve.csv", 
    parse_dates={"Date": ["Year", "Month", "Day"]},
    index_col="Date",
)
logger.info(f"reserve: \n{reserve} \nreserve.columns: \n{reserve.columns}")

y = reserve.loc[:, "Unemployment Rate"].dropna().to_period("M")
logger.info(f"y: \n{y}")

df = pd.DataFrame({
    "y": y,
    "y_lag_1": y.shift(1),
    "y_lag_2": y.shift(2),
})
logger.info(f"df: \n{df}")


# ------------------------------
# lag plots
# ------------------------------



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
