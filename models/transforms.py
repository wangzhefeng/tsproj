# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transforms.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101715
# * Description : 序列数据预处理类函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pysindy as ps
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def SeriesDataNormalize(series):
    """
    数据序列归一化函数, 受异常值影响

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 归一化对象
        normalized: 归一化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(series)
    normalized = scaler.transform(series)

    return scaler, normalized


def SeriesDataStandardScaler(series):
    """
    数据序列标准化函数, 不受异常值影响

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 标准化对象
        normalized: 标准化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(series)
    normalized = scaler.transform(series)

    return scaler, normalized




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
