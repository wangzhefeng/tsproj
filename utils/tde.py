# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tde.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-26
# * Version     : 1.0.052616
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
import re

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int, return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning

    Args:
        series: time series as pd.Series
        n_lags: number of past values to used as explanatory variables
        horizon: how many values to forecast
        return_Xy: whether to return the lags split from future observations

    Return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)
    # series name
    if series.name is None:
        name = 'Series'
    else:
        name = series.name
    # create features
    n_lags_iter = list(range(n_lags, -horizon, -1))
    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    # features rename
    df.columns = [
        f'{name}(t-{j - 1})' if j > 0 else f'{name}(t+{np.abs(j) + 1})'
        for j in n_lags_iter
    ]
    df.columns = [re.sub('t-0', 't', x) for x in df.columns]
    # 返回 pandas.Dataframe
    if not return_Xy:
        return df
    # future features
    is_future = df.columns.str.contains('\\+')
    # feature split
    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y




# 测试代码 main 函数
def main():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    logger.info(f"series: \n{series}")
    df = time_delay_embedding(series, n_lags=3, horizon=1, return_Xy=False)
    X, y = time_delay_embedding(series, n_lags=3, horizon=1, return_Xy=True)
    logger.info(f"df: \n{df}")
    logger.info(f"X: \n{X} \ny: \n{y}")

if __name__ == "__main__":
    main()
