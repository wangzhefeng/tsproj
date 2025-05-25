# -*- coding: utf-8 -*-

# ***************************************************
# * File        : standard.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from sklearn.preprocessing import StandardScaler
from math import sqrt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def stan_series(series):
    """
    时间序列标准化
    """
    # 准备数据
    values = series.values
    values = values.reshape((len(values), 1))

    # 定义标准化模型
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    
    # 开始标准化，打印前五行
    normalized = scaler.transform(values)
    # for i in range(5):
    #     print(normalized[i])
    # 逆标准化数据
    inversed = scaler.inverse_transform(normalized)
    # for i in range(5):
    #     print(inversed[i])

    return normalized, inversed


class StandardScaler():
    """
    标准化
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



# 测试代码 main 函数
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    # ------------------------------
    # 判断数据是否适用标准化
    # ------------------------------ 
    # 数据读取
    series = pd.read_csv(
        "E:/projects/timeseries_forecasting/tsproj/dataset/daily-minimum-temperatures-in-me.csv",
        header = 0,
        index_col = 0,
        # parse_dates = [0],
        # date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
    )
    print(series.head())
    # 根据数据分布图判断数据是否服从正太分布
    # series.hist()
    # plt.show()
    # ------------------------------
    # 时间序列标准化
    # ------------------------------
    normalized, inversed = stan_series(series)
    print(normalized)
    print(inversed) 
    
if __name__ == "__main__":
    main()
