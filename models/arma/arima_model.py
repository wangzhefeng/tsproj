# -*- coding: utf-8 -*-

# ***************************************************
# * File        : arima_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090823
# * Description : ARIMA 模型
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings
from random import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 平稳性检验(AD检验)
from statsmodels.tsa.stattools import acf, adfuller, pacf
# 模型分解
from statsmodels.tsa.seasonal import seasonal_decompose
# ARIMA 模型
from statsmodels.tsa.SARIMAX import ARIMA

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = 15, 6
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def stationarity_test(ts):
    """
    ADFuller 平稳性检验

    Args:
        ts (_type_): _description_
    """
    # rolling statistics
    rollmean = pd.Series.rolling(ts, window = 12).mean()
    rollstd = pd.Series.rolling(ts, window = 12).std()

    orig = plt.plot(ts, color = "blue", label = "Original")
    mean = plt.plot(rollmean, color = "red", label = "Rolling mean")
    std = plt.plot(rollstd, color = "black", label = "Rolling std")
    plt.legend(loc = "best")
    plt.title("Rolling mean & Standard Deviation")
    plt.show()

    # Dickey Fuller test
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(ts, autolag = "AIC")
    dfountput = pd.Series(dftest[0:4], 
                        index = ["Test Statistic", 
                                "p-value", 
                                "#lag used", 
                                "Number of observation used"])
    for key, value in dftest[4].items():
        dfountput["Critical Value(%s)" % key] = value


def arima_performance(data, order1):
    model = ARIMA(data, order = order1)
    results_arima = model.fit(disp = -1)
    results_arima_value = results_arima.fittedvalues
    results_future = results_arima.forecast(7)
    return results_arima_value, results_future


def arima_plot(data, results_arima_value):
    plt.plot(data)
    plt.plot(results_arima_value, color = "red")
    plt.title("RSS: %.4f" % sum((results_arima_value) ** 2))


def add_season(ts_recover_trend, startdate):
    ts2_season = ts2_season
    values = []
    low_conf_values = []


# 测试代码 main 函数
def main():
    data = [x + random() for x in range(1, 100)]
    model = ARIMA(data, order = (1, 1, 1))
    model_fit = model.fit(disp = True)
    y_hat = model_fit.predict(len(data), len(data), type = "levels")
    print(y_hat)

if __name__ == "__main__":
    main()
