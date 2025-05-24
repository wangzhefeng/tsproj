# -*- coding: utf-8 -*-

# ***************************************************
# * File        : stationarity_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090822
# * Description : 时间序列平稳性检验
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest, PPTest, KPSSTest

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

"""
* 自相关图平稳检测：ACF 图、PACF 图，如果图看起来不相对平稳，模型可能需要差分项.
* 差分平稳：ARIMAs that include differencing (i.e., d > 0) assume that the data becomes stationary after differencing. This is called difference-stationary. 
* ADF test(Augmented Dickey-Fuller test)
* `pmdarima.auto_arima` 能够自动确定适当的差分项.
"""

def get_acf(series: pd.Series, is_visual: bool = True):
    """
    计算 ACF，ACF 可视化

    Args:
        series (pd.Series): _description_
        is_visual (bool): _description_
    """
    series = pm.c(series.values) if isinstance(series, pd.Series) else series
    acf_value = pm.acf(series)

    if is_visual:
        pm.plot_acf(series)
     
    return acf_value


def get_pacf(series: pd.Series, is_visual: bool = True):
    """
    计算 PACF，PACF 可视化

    Args:
        series (pd.Series): _description_
        is_visual (bool): _description_
    """
    series = pm.c(series.values) if isinstance(series, pd.Series) else series
    pacf_value = pm.pacf(series)

    if is_visual:
        pm.plot_pacf(series)
    
    return pacf_value


def ADF_test(series):
    """
    ADF test
    """
    adf_test = ADF_test(alpha = 0.05)
    p_val, should_diff = adf_test.should_diff(series)
    
    return p_val, should_diff


def PP_test(series):
    """
    PP tset
    """
    pp_test = PP_test(alpha = 0.05)
    p_val, should_diff = pp_test.should_diff(series)
    
    return p_val, should_diff


def KPSS_test(series):
    """
    KPSS test
    """
    kpss_test = KPSS_test(alpha = 0.05)
    p_val, should_diff = kpss_test.should_diff(series)
    
    return p_val, should_diff


def stationarity_test(series):
    """
    时间序列平稳性检测
    """
    from pmdarima.arima.utils import ndiffs

    # Estimate the number of differences using an ADF test:
    n_adf = ndiffs(series, test = 'adf')  # -> 0

    # Or a KPSS test (auto_arima default):
    n_kpss = ndiffs(series, test = 'kpss')  # -> 0

    # Or a PP test:
    n_pp = ndiffs(series, test = 'pp')  # -> 0
    
    assert n_adf == n_kpss == n_pp == 0


def stationarity_test(timeseries, window: int = 12):
    """
    平稳性检验
    """
    plt.rcParams["figure.figsize"] = 15, 6
    # determing rolling statistics
    rolmean = timeseries.rolling(window = window).mean()
    rolstd = timeseries.rolling(window = window).std()
    # rollmean = pd.Series.rolling(ts, window = 12).mean()
    # rollstd = pd.Series.rolling(ts, window = 12).std()
    
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
]


# 测试代码 main 函数
def main():
    ts = None
    stationarity_test(ts)

if __name__ == "__main__":
    main()
