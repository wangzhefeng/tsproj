# -*- coding: utf-8 -*-

# ***************************************************
# * File        : HoldOutCV.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-10
# * Version     : 0.1.091017
# * Description : 时间序列 Hold-Out 交叉验证
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def train_test_split(series, train_rate: float = 0.66):
    """
    时间序列数据分割
    """
    X = series.values
    train_size = int(len(X) * train_rate)

    train, test = X[0:train_size], X[train_size:len(X)]
    # print('Observations: %d' % (len(X)))
    # print('Training Observations: %d' % (len(train)))
    # print('Testing Observations: %d' % (len(test)))

    # data plot
    # plt.plot(train)
    # plt.plot([None for i in train] + [x for x in test])
    # plt.show()

    return train, test




# 测试代码 main 函数
def main():
    import pandas as pd

    # 数据读取
    series = pd.read_csv(
        "D:/projects/timeseries_forecasting/tsproj/dataset/sunspots.csv",
        header = 0,
        index_col = 0,
        # parse_dates = [0],
        # date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
    )
    print(series.head())
    
    # 数据分割
    train, test = train_test_split(series, train_rate = 0.66)

if __name__ == "__main__":
    main()
