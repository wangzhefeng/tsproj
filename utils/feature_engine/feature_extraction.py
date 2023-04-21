# -*- coding: utf-8 -*-


# ***************************************************
# * File        : feature_extraction.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-07
# * Version     : 0.1.120719
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
from scipy import stats
import statsmodels.tsa.stattools as ts


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Covariance between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_

    Returns:
        float: _description_
    """
    cov_xy = np.cov(x, y)[0][1]

    return cov_xy


def co_integration(x: np.ndarray, y: np.ndarray):
    """
    Co-intergration test between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    r, _, _ = ts.coint(x, y)

    return r


def correlation(x: np.ndarray, y: np.ndarray, method: str = "kendall"):
    """
    Correlation between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        method (str, optional): _description_. Defaults to "kendall".

    Returns:
        _type_: _description_
    """
    assert method in ["pearson", "spearman", "kendall"]

    corr, p_value = stats.kendalltau(x, y)

    return corr




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

