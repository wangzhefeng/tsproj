# -*- coding: utf-8 -*-


# ***************************************************
# * File        : plot_windows.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-11
# * Version     : 0.1.031115
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from warnings import simplefilter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_windows(y, cv):
    """
    Generate windows

    Args:
        y (_type_): _description_
        cv (_type_): _description_
    """
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    
    return train_windows, test_windows


def plot_windows(y, train_windows, test_windows, title = ""):
    """
    Visualize training and test windows

    Args:
        y (_type_): _description_
        train_windows (_type_): _description_
        test_windows (_type_): _description_
        title (str, optional): _description_. Defaults to "".
    """
    simplefilter("ignore", category = UserWarning)

    def get_y(length, split):
        """
        Create a constant vecotr based on the split for y-axis.

        Args:
            length (_type_): _description_
            split (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.ones(length) * split
    
    # split params
    n_splits = len(train_windows)  # TODO
    n_timepoints = len(y)  # TODO
    len_test = len(test_windows[0])  # TODO

    # plot params
    train_color, test_color = sns.color_palette("colorblind")[:2]
    fig, ax = plt.subplots(figsize = plt.figaspect(0.3))

    for i in range(n_splits):
        # TODO
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(np.arange(n_timepoints), get_y(n_timepoints, i), marker = "o", c = "lightgray")
        ax.plot(train, get_y(len(train), i), marker = "o", c = train_color, label = "Window")
        ax.plot(test, get_y(len_test, i), marker = "o", c = test_color, label = "Forecasting horizon")

        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))
        ax.set(
            title = title,
            ylabel = "Window number",
            xlabel = "Time",
            xticklabels = y.index,
        )
        # remove duplicate labels/handlers
        handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
        ax.legend(handles, labels);




__all__ = [
    plot_windows,
    get_windows,
]


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()







