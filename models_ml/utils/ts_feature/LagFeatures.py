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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sklearn import base

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LagFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, 
                 col, 
                 # groupCol, 
                 numLags, 
                 dropna = False):
        self.col = col
        # self.groupCol = groupCol
        self.numLags = numLags
        self.dropna = dropna

    def fit(self, X, y = None):
        self.X = X
        return self

    def transform(self, X):
        tmp = self.X.copy()
        for i in range(1, self.numLags + 1):
            # tmp[str(i) + "_Week_Ago" + "_" + self.col] = tmp.groupby([self.groupCol])[self.col].shift(i)
            tmp[f"{self.col}(t-{i})"] = tmp[self.col].shift(i)
        
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop = True)
        
        return tmp


def time_delay_embedding(series: pd.Series, n_lags: int = 1, horizon: int = 0, dropna: bool = False):
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
    # 特征生成
    n_lags_iter = list(range(n_lags, -horizon, -1))
    X = [series.shift(i) for i in n_lags_iter]
    X = pd.concat(X, axis = 1)
    # 删除缺失值
    if dropna:
        X = X.dropna()
    # 重命名 
    X.columns = [
        f"{name}(t-{j})" if j > 0 else f"{name}(t+{np.abs(j)+1})"
        for j in n_lags_iter
    ]
    X.reset_index(inplace=True)
    
    return X


# TODO
def gen_lag_features(data, cycle):
    """
    时间序列滞后性特征
        - 二阶差分
    Args:
        data ([type]): 时间序列
        cycle ([type]): 时间序列周期
    """
    # 序列平稳化, 季节性差分
    series_diff = data.diff(cycle)
    series_diff = series_diff[cycle:]
    # 监督学习的特征
    for i in range(cycle, 0, -1):
        series_diff["t-" + str(i)] = series_diff.shift(i).values[:, 0]
    series_diff["t"] = series_diff.values[:, 0]
    series_diff = series_diff[cycle + 1:]

    return series_diff

# TODO
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
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
    if dropnan:
        agg.dropna(inplace = True)

    return agg





# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    # data
    df = pd.DataFrame({
        "ts": pd.date_range(start="2024-11-14 00:00:00", end="2024-11-15 00:46:00", freq="15min"),
        "unique_id": range(100),
        "load": np.random.randn(100),
        # "load2": np.random.randn(100),
    })
    # df.set_index("ts", drop=True,inplace=True)
    print(df)
    print("-" * 80)
    # ------------------------------
    # LagFeatures 
    # ------------------------------
    # processer
    model = LagFeatures(col = "load", numLags = 3, dropna = True)
    # processing
    df_lags = model.fit_transform(df)
    
    df["load(t-1)"] = df["ts"].map(df_lags.set_index("ts")["load(t-1)"])
    df["load(t-2)"] = df["ts"].map(df_lags.set_index("ts")["load(t-2)"])
    df["load(t-3)"] = df["ts"].map(df_lags.set_index("ts")["load(t-3)"])
    
    with pd.option_context("display.max_columns", None):
        print(df)
        print("-" * 80)
    # ------------------------------
    # time_delay_embedding
    # ------------------------------
    X = time_delay_embedding(series=df.set_index("ts")["load"], n_lags=3, horizon=0, dropna=True) 
    # df["load_lag_11"] = df["ts"].map(X.set_index("ts")["load(t+1)"])
    df["load(t-1)"] = df["ts"].map(X.set_index("ts")["load(t-1)"])
    df["load(t-2)"] = df["ts"].map(X.set_index("ts")["load(t-2)"])
    df["load(t-3)"] = df["ts"].map(X.set_index("ts")["load(t-3)"])
    
    with pd.option_context("display.max_columns", None):
        print(df)
        print("-" * 80)
    """
    # ------------------------------
    # series_to_supervised
    # ------------------------------
    # 一步式单变量，多测一 
    # 单变量 步长为 1 的监督学习
    values = [float(x) for x in range(10)]
    data = series_to_supervised(data = values, n_in = 1, n_out = 1)
    print(data)

    # 单变量 步长为 3 的监督学习
    values = [x for x in range(10)]
    data = series_to_supervised(data = values, n_in = 3, n_out = 1)
    print(data)
    # 一步式多变量，一测一
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    data = series_to_supervised(data = values, n_in = 1, n_out = 1)
    print(data)
    # 多步式多变量，一测多
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    data = series_to_supervised(values, 1, 2)
    print(data)
    # 多步式单变量，多测多
    # 单变量步长为 2 预测两步
    values = [x for x in range(10)]
    data = series_to_supervised(values, 2, 2)
    print(data)
    """

if __name__ == "__main__":
    main()
