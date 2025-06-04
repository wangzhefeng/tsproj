# -*- coding: utf-8 -*-

# ***************************************************
# * File        : cv_plot.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-04
# * Version     : 1.0.060417
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
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit
cmap_cv = plt.cm.coolwarm
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
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


def plot_cv_indices(cv, n_splits, X, y, date_col = None):
    """
    Create a sample plot for indices of a cross-validation object.
    """
    fig, ax = plt.subplots(1, 1, figsize = (11, 7))
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)), 
            [ii + 0.5] * len(indices),
            c = indices, 
            marker = '_', 
            lw = 10, 
            cmap = cmap_cv,
            vmin = -0.2, vmax = 1.2,
        )
    # Formatting
    yticklabels = list(range(n_splits))
    if date_col is not None:
        tick_locations  = ax.get_xticks()
        tick_dates = [" "] + date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() + [" "]
        tick_locations_str = [str(int(i)) for i in tick_locations]
        new_labels = ['\n\n'.join(x) for x in zip(list(tick_locations_str), tick_dates) ]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)
    ax.set(
        yticks = np.arange(n_splits) + 0.5,
        yticklabels = yticklabels,
        xlabel = "Sample index",
        ylabel = "CV iteration",
        ylim = [n_splits+0.2, -0.2]
    )
    ax.legend(
        [Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
        ['Testing set', 'Training set'], loc=(1.02, .8)
    )
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    plt.show();



# 测试代码 main 函数
def main():
    n_points = 100
    n_splits = 5
    X = np.random.randn(n_points, 10)
    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack([
        [i] * int(n_points * perc)
        for i, perc in enumerate(percentiles_classes)
    ])
    logger.info(f"X: \n{X}")
    logger.info(f"y: \n{y}")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    plot_cv_indices(tscv, n_splits, X, y)

if __name__ == "__main__":
    main()
