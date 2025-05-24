# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LagFeatures.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-11-15
# * Version     : 1.0.111513
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

import numpy as np
import pandas as pd
from sklearn import base

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LagFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, target: str, group_col: str = None, numLags: int = 1, numHorizon: int = 0, dropna: bool = False):
        """
        Time delay embedding.
        Time series for supervised learning.

        Args:
            target (str): _description_
            group_col (str, optional): _description_. Defaults to None.
            numLags (int, optional): number of past values to used as explanatory variables.. Defaults to 1.
            numHorizon (int, optional): how many values to forecast. Defaults to 0.
            dropna (bool, optional): _description_. Defaults to False.
        """
        self.target = target
        self.group_col = group_col
        self.numLags = numLags
        self.numHorizon = numHorizon
        self.dropna = dropna

    def fit(self, X, y = None):
        self.X = X
        return self

    def transform(self, X):
        # 滞后特征构造
        tmp = self.X.copy()
        # for i in range(1, self.numLags + 1):
        for i in range(self.numLags, -self.numHorizon, -1):
            if self.group_col is None:
                if i == 0:
                    tmp[f"{self.target}(t)"] = tmp[self.target].shift(i)
                elif i < 0:
                    tmp[f"{self.target}(t+{abs(i)})"] = tmp[self.target].shift(i)
                else:
                    tmp[f"{self.target}(t-{i})"] = tmp[self.target].shift(i)
            else:
                if i == 0:
                    tmp[f"{self.target}(t)"] = tmp.groupby(self.group_col)[self.target].shift(i)
                elif i < 0:
                    tmp[f"{self.target}(t+{abs(i)})"] = tmp.groupby(self.group_col)[self.target].shift(i)
                else:
                    tmp[f"{self.target}(t-{i})"] = tmp.groupby(self.group_col)[self.target].shift(i)
        # 缺失值处理
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop = True)
        
        return tmp


def lag_features(df: pd.DataFrame, target: str, group_col: str = None, numLags: int = 3, numHorizon: int = 0, dropna: bool = False):
    """
    Time delay embedding.
    Time series for supervised learning.

    Args:
        target (str): _description_
        group_col (str, optional): _description_. Defaults to None.
        numLags (int, optional): number of past values to used as explanatory variables.. Defaults to 1.
        numHorizon (int, optional): how many values to forecast. Defaults to 0.
        dropna (bool, optional): _description_. Defaults to False.
    """
    # 滞后特征构造
    tmp = df.copy()
    # for i in range(1, self.numLags + 1):
    for i in range(numLags, -numHorizon, -1):
        if group_col is None:
            if i <= 0:
                tmp[f"{target}(t+{abs(i)+1})"] = tmp[target].shift(i)
            else:
                tmp[f"{target}(t-{numLags + 1 - i})"] = tmp[target].shift(i)
        else:
            if i <= 0:
                tmp[f"{target}(t+{abs(i)+1})"] = tmp.groupby(group_col)[target].shift(i)
            else:
                tmp[f"{target}(t-{numLags + 1 - i})"] = tmp.groupby(group_col)[target].shift(i)
    # 缺失值处理
    if dropna:
        tmp = tmp.dropna()
        tmp = tmp.reset_index(drop = True)
    
    return tmp


# TODO
def series_to_supervised(data, n_in = 1, n_out = 1, dropna = False):
    """
    Frame a time series as a supervised learning dataset.

    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    # drop rows with NaN values
    if dropna:
        agg.dropna(inplace = True)

    return agg





# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    # data
    df1 = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-17 00:00:00", end="2024-11-17 09:00:00", freq="1h"),
        "unique_id": [1] * 10,
        "load": range(1, 11),
        "load1": range(1, 11),
        # "load2": np.random.randn(100),
    })
    print(df1)
    print("-" * 80)
    
    # df2 = pd.DataFrame({
    #     "ds": pd.date_range(start="2024-11-17 00:00:00", end="2024-12-14 23:00:00", freq="1h"),
    #     "unique_id": [1] * 336 + [2] * 336,
    #     "load": range(672),
    #     # "load2": np.random.randn(100),
    # })
    # print(df2)
    # print("-" * 80)
    
    """
    # ------------------------------
    # LagFeatures 
    # ------------------------------
    # processer
    lag_pro = LagFeatures(target = "load", group_col=None, numLags=24, numHorizon=0, dropna = True)
    # processing
    df_lags = lag_pro.fit_transform(df1)
    with pd.option_context("display.max_columns", None):
        print(df_lags)
        print("-" * 80)
    """
    # ------------------------------
    # time_delay_embedding
    # ------------------------------
    # df_lags = lag_features(df1, target="load", group_col="unique_id", numLags=0, numHorizon=0, dropna=False)
    # with pd.option_context("display.max_columns", None, "display.max_rows", None):
    #     print(df_lags)
    #     print("-" * 80)
    
    df_lags = lag_features(df1, target="load", group_col="unique_id", numLags=5, numHorizon=0, dropna=True)
    with pd.option_context("display.max_columns", None, "display.max_rows", None):
        print(df_lags)
        print("-" * 80)
    
    df_lags = lag_features(df1, target="load", group_col="unique_id", numLags=0, numHorizon=5, dropna=False)
    df_lags = df_lags.iloc[-5:, ]
    with pd.option_context("display.max_columns", None, "display.max_rows", None):
        print(df_lags)
        print("-" * 80)
    """
    # ------------------------------
    # series_to_supervised
    # ------------------------------
    # 一步式单变量，多测一
    # -----------------
    # 单变量: 步长为 1 的监督学习
    values = [float(x) for x in range(10)]
    print(values)
    data = series_to_supervised(data = values, n_in = 1, n_out = 1, dropna=True)
    print(data)
    print("-" * 80)
    
    # 单变量: 步长为 3 的监督学习
    values = [x for x in range(10)]
    print(values)
    data = series_to_supervised(data = values, n_in = 3, n_out = 1, dropna = True)
    print(data)
    print("-" * 80)
    
    # 一步式多变量，一测一
    # -----------------
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    print(values)
    data = series_to_supervised(data = values, n_in = 1, n_out = 1, dropna=True)
    print(data)
    print("-" * 80)

    # 多步式多变量，一测多
    # -----------------
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    print(values)
    data = series_to_supervised(values, n_in=1, n_out=2, dropna=False)
    print(data)
    print("-" * 80)
    
    # 多步式单变量，多测多
    # -----------------
    # 单变量步长为 2 预测两步
    values = [x for x in range(10)]
    print(values)
    data = series_to_supervised(values, n_in=2, n_out=2, dropna=True)
    print(data)
    """
    
if __name__ == "__main__":
    main()
