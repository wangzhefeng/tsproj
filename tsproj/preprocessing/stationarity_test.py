# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams["figure.figsize"] = 15, 6
from statsmodels.tsa.stattools import adfuller


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def stationarity_test(timeseries, window: int = 12):
    """
    季节性检测

    :param timeseries: _description_
    :type timeseries: _type_
    :param window: _description_, defaults to 12
    :type window: int, optional
    """
    # determing rolling statistics
    rolmean = timeseries.rolling(window = window).mean()
    rolstd = timeseries.rolling(window = window).std()
    
    # plot rolling statistics
    orig = plt.plot(timeseries, color = "blue", label = "Original")
    mean = plt.plot(rolmean, color = "red", label = "Rolling Mean")
    std = plt.plot(rolstd, color = "black", label = "Rolling Std")
    plt.legend(loc = "best")
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block = False)

    # perform Dickey-Fuller test
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag = "AIC")
    dfoutput = pd.Series(
        dftest[0:4], 
        index = [
            "Test Statistic", 
            "p-value", 
            "#Lags Used", 
            "Number of Observations Used"
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Cirtical Value (%s)" % key] = value
    print(dfoutput)





__all__ = [
    stationarity_test,
    remove_trend,
]


# 测试代码 main 函数
def main():
    ts = None
    stationarity_test(ts)
    remove_trend(ts, method = "log", chart = False)


if __name__ == "__main__":
    main()




