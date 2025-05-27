# -*- coding: utf-8 -*-

# ***************************************************
# * File        : difference.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090823
# * Description : 时间序列差分、反差分
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pmdarima as pm

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def difference_v1(series, lag = 1):
    """
    对序列差分
    """
    diff = []
    for i in range(lag, len(series)):
        value = series[i] - series[i - lag]
        diff.append(value)

    return diff


def difference_v2(series, period, axis: int = 0):
    """
    # 1 阶差分、1步差分
    pandas.DataFrame.diff(periods = 1, axis = 0)

    # 2 步差分
    pandas.DataFrame.diff(periods = 2, axis = 0)

    # k 步差分
    pandas.DataFrame.diff(periods = k, axis = 0)

    # -1 步差分
    pandas.DataFrame.diff(periods = -1, axis = 0)
    """
    pass

def difference(series: np.array, lag: int = 1, differences: int = 1):
    """
    差分运算

    Args:
        series (np.array): 原始时间序列
        lag (int, optional): 差分阶数. Defaults to 1.
        differences (int, optional): 差分步数. Defaults to 1.

    Returns:
        _type_: _description_
    """
    series_diff = pm.utils.diff(series, lag = lag, differences = differences)
    
    return series_diff


def inverse_difference_v1(series, diff_series):
    """
    对序列反差分
    """
    inverted_series = [series[i] + diff_series[i] for i in range(len(diff_series))]
    
    return inverted_series


def inverse_difference(series_diff: np.array, lag: int = 1, differences: int = 1):
    """
    对序列反差分
    """
    pass


# 测试代码 main 函数
def main():
    # ------------------------------
    # 自定义
    # ------------------------------
    series = [i + 1 for i in range(20)]
    print(series)
    print(len(series))
    
    diff = difference_v1(series, lag=2)
    print(diff)
    print(len(diff))
    
    inversed = inverse_difference_v1(series, diff)
    print(inversed)
    print(len(inversed))
    # ------------------------------
    # pmdarima
    # ------------------------------
    x = pm.c(10, 4, 2, 9, 34)
    print(x)
    
    x_diff = pm.utils.diff(x, lag = 1, differences = 1)
    print(x_diff)
    
    x_diff_inv = pm.utils.diff_inv(x, lag = 1, differences = 1)
    print(x_diff_inv)

if __name__ == "__main__":
    main()
