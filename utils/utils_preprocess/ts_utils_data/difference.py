# -*- coding: utf-8 -*-


# ***************************************************
# * File        : diff.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-19
# * Version     : 0.1.111901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def difference(series, interval = 1):
    """
    对序列差分

    Args:
        series (_type_): _description_
        interval (int, optional): _description_. Defaults to 1.
    """
    diff = list()
    for i in range(interval, len(series)):
        value = series[i] - series[i - interval]
        diff.append(value)
    
    return diff


def inverse_diff_func(last_ob, value):
    """
    反差分计算

    Args:
        last_ob (_type_): _description_
        value (_type_): _description_
    """
    return value + last_ob


def inverse_difference(series, diff_series):
    """
    对序列反差分

    Args:
        series (_type_): _description_
        diff_series (_type_): _description_

    Returns:
        _type_: _description_
    """
    inverted_series = [
        inverse_diff_func(series[i], diff_series[i]) for i in range(len(diff_series))
    ]
    return inverted_series




# 测试代码 main 函数
def main():
    series = [i + 1 for i in range(20)]
    diff = difference(series)
    print(diff)
    inversed = inverse_difference(series, diff)


if __name__ == "__main__":
    main()

