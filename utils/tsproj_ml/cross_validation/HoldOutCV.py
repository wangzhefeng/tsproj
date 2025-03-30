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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def Holdout_train_test_split(data_X, data_Y, data_length: int, step_length: int):
    """
    数据集分割
    """
    X_train = data_X.iloc[-data_length:-step_length]
    Y_train = data_Y.iloc[-data_length:-step_length]
    X_test = data_X.iloc[-step_length:]
    Y_test = data_Y.iloc[-step_length:]

    return X_train, Y_train, X_test, Y_test


# TODO
def Kfold_train_test_split(data_X, 
                           data_Y, 
                           n_windows: int, 
                           data_length: int, 
                           step_length: int, 
                           train_start: int, 
                           train_end: int, 
                           valid_start: int, 
                           valid_end: int):
    for step in range(n_windows):
        # 数据分割索引构造
        valid_end   = -1        + (-step_length) * step
        valid_start = valid_end + (-step_length) + 1
        train_end   = valid_start
        train_start = valid_end + (-data_length) + 1
        # 数据分割: 训练集、测试集
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if valid_end == -1:
            X_test = data_X.iloc[valid_start:]
            Y_test = data_Y.iloc[valid_start:]
            logger.info(f"split index: {train_start}:{train_end}, {valid_start}:{''}")
        else:
            X_test = data_X.iloc[valid_start:(valid_end+1)]
            Y_test = data_Y.iloc[valid_start:(valid_end+1)]
            logger.info(f"split index: {train_start}:{train_end}, {valid_start}:{valid_end+1}")

        logger.info(f"length of X_train: {len(X_train)}, length of Y_train: {len(Y_train)}")
        logger.info(f"length of X_test: {len(X_test)}, length of Y_test: {len(Y_test)}")

    return X_train, Y_train, X_test, Y_test




# 测试代码 main 函数
def main():
    import pandas as pd

    # 数据读取
    series = pd.read_csv(
        "E:/projects/timeseries_forecasting/tsproj/dataset/sunspots.csv",
        header = 0,
        index_col = 0,
        # parse_dates = [0],
        # date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
    )
    print(series.head())
    
    # 数据分割
    

if __name__ == "__main__":
    main()
