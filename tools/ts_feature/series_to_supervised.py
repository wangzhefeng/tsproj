# -*- coding: utf-8 -*-

# ***************************************************
# * File        : series_to_supervised.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102023
# * Description : 时间序列转换为监督学习格式
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    # 一步式单变量，多测一 
    # ------------------------------
    # 单变量 步长为 1 的监督学习
    values = [float(x) for x in range(10)]
    data = series_to_supervised(data = values, n_in = 1, n_out = 1)
    print(data)

    # 单变量 步长为 3 的监督学习
    values = [x for x in range(10)]
    data = series_to_supervised(data = values, n_in = 3, n_out = 1)
    print(data)
    # ------------------------------
    # 一步式多变量，一测一
    # ------------------------------
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    data = series_to_supervised(data = values, n_in = 1, n_out = 1)
    print(data)
    # ------------------------------
    # 多步式多变量，一测多
    # ------------------------------
    raw = pd.DataFrame()
    raw['ob1'] = [x for x in range(10)]
    raw['ob2'] = [x for x in range(50, 60)]
    values = raw.values
    data = series_to_supervised(values, 1, 2)
    print(data)
    # ------------------------------
    # 多步式单变量，多测多
    # ------------------------------
    # 单变量步长为 2 预测两步
    values = [x for x in range(10)]
    data = series_to_supervised(values, 2, 2)
    print(data) 

if __name__ == "__main__":
    main()
