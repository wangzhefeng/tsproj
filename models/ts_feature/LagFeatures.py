# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LagFeatures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042415
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LagFeatures:
    
    def __init__(self, 
                 data, 
                 datetime_format: str = '%Y-%m-%d %H:%M:%S',
                 window_length: int = 7, 
                 lags: int = 1) -> None:
        self.data = data
        self.datetime_format = datetime_format
        self.window_length = window_length
        self.lags = lags

    def features(self, raw_feature, new_feature):
        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"], format = self.datetime_format)
        self.data[new_feature] = self.data[raw_feature].shift(self.lags)
        data_columns = ["Datetime", raw_feature, new_feature]
        self.data = self.data[data_columns]


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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
