# -*- coding: utf-8 -*-

# ***************************************************
# * File        : find_trend.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-24
# * Version     : 1.0.052422
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

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_trend(df, model: str="multiplicative"):
    """
    趋势分析

    Args:
        df (_type_): 待分析数据集
        model (str, optional): 时间序列分解模型. 
            Defaults to "multiplicative". 
            Option: "multiplicative", "additive"
    """
    # series decomposition
    if model == "multiplicative":
        decomposition = seasonal_decompose(df, model='multiplicative', period =50)
    elif model == "additive":
        decomposition = seasonal_decompose(df, model='additive', period =50)
    # series trend
    trend = decomposition.trend
    # plot
    trend.plot()
    plt.title("Series Trend")
    plt.show();




# 测试代码 main 函数
def main():
    import pandas as pd

    df = pd.read_csv("./dataset/wind_dataset.csv", index_col=["DATE"], parse_dates=["DATE"])
    df = df["WIND"]
    df = df.ffill()
    df = df.bfill()
    print(df.head())
    df.plot()
    plt.show()
    
    plot_trend(df, model="additive")

if __name__ == "__main__":
    main()
