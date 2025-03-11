# -*- coding: utf-8 -*-

# ***************************************************
# * File        : normalization.py
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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sklearn.preprocessing import MinMaxScaler

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def norm_series(series):
    """
    时间序列归一化
    """
    # 准备归一化数据
    values = series.values
    values = values.reshape((len(values), 1))
    
    # 定义缩放范围(0, 1)
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    
    # 归一化数据集
    normalized = scaler.transform(values)
    # for i in range(5):
    #     print(normalized[i])
    
    # 逆变换并打印前5行
    inversed = scaler.inverse_transform(normalized)
    # for i in range(5):
    #     print(inversed[i])

    return normalized, inversed




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
    # 时间序列归一化
    # ------------------------------
    normalized, inversed = norm_series(series)
    print(normalized)
    print(inversed) 

if __name__ == "__main__":
    main()
