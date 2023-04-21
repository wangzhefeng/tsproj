# -*- coding: utf-8 -*-


# ***************************************************
# * File        : time_delay_embedding.py
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
import pandas as pd


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int):
    """
    Time delay embedding
    Time series for supervised learning

    Args:
        series (pd.Series): time series as pd.Series
        n_lags (int): number of past values to used as explanatory variables
        horizon (int): how many values to forecast
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = "Series"
    else:
        name = series.name
    
    n_lags_iter = list(range(n_lags, -horizon, -1))

    X = [series.shift(i) for i in n_lags_iter]
    X = pd.concat(X, axis = 1).dropna()
    X.columns = [
        f"{name}(t-{j-1})"
        if j > 0 else f"{name}(t+{np.abs(j)+1})"
        for j in n_lags_iter
    ]
    
    return X




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

