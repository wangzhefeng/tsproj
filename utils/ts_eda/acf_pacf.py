# -*- coding: utf-8 -*-

# ***************************************************
# * File        : acf_pacf.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090822
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
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_acf(series: pd.Series, is_visual: bool = True):
    """
    计算 ACF，ACF 可视化
    """
    import pmdarima as pm

    series = pm.c(series.values) if isinstance(series, pd.Series) else series
    acf_value = pm.acf(series)

    if is_visual:
        pm.plot_acf(series)
     
    return acf_value


def get_pacf(series: pd.Series, is_visual: bool = True):
    """
    计算 PACF，PACF 可视化
    """
    import pmdarima as pm

    series = pm.c(series.values) if isinstance(series, pd.Series) else series
    pacf_value = pm.pacf(series)

    if is_visual:
        pm.plot_pacf(series)
    
    return pacf_value


def get_acf_pacf(series: pd.Series, nlags: int = 3):
    """
    ACF，PACF
    """
    import statsmodels.api as sm
    
    lag_acf = sm.tsa.stattools.acf(series, nlags = nlags)
    lag_pacf = sm.tsa.stattools.pacf(series, nlags = nlags, method = "ols")

    fig = plt.figure(figsize = (12, 8))
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y = 0, linestyle = "--", color = "gray")
    # plt.axhline(y = - 1.96 / np.sqrt(len(series)), linestyle = "", color = "gray")
    # plt.axhline(y = 1.96 / np.sqrt(len(series)), linestyle = "", color = "gray")
    plt.title("Autocorrelation Function")

    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y = 0, linestyle = "--", color = "gray")
    # plt.axhline(y = - 1.96 / np.sqrt(len(series)), linestyle = "", color = "gray")
    # plt.axhline(y = 1.96 / np.sqrt(len(series)), linestyle = "", color = "gray")
    plt.title("Partial Autocorrelation Function")

    plt.tight_layout()
    plt.show()

    return lag_acf, lag_pacf


def plot_acf_pacf(series_list: List[pd.Series], series_names: List[str], nlags: int = 3):
    """
    plot ACF, PACF

    Args:
        series_list (List[pd.Series]): _description_
        series_names (List[str]): _description_
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, ax = plt.subplots(len(series_list), 2, figsize = (12, 8))
    fig.subplots_adjust(hspace = 0.5)
    for i, series in enumerate(series_list):
        plot_acf(series, lags=nlags,ax = ax[i][0])
        ax[i][0].set_title(f'ACF({series_names[i]})')
        plot_pacf(series, lags=nlags, ax = ax[i][1])
        ax[i][1].set_title(f'PACF({series_names[i]})')
    plt.tight_layout()
    plt.show();




# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    import numpy as np
    # 白噪声
    white_noise = np.random.standard_normal(size = 1000)

    # 随机游走
    x = np.random.standard_normal(size = 1000)
    random_walk = np.cumsum(x)
    # ------------------------------
    # pmdarima
    # ------------------------------
    # import pmdarima as pm
    # x = pm.c(1, 2, 3, 4, 5, 6, 7)
    # print(x)

    # acf_value = get_acf(x)
    # print(acf_value)

    # pacf_value = get_pacf(x)
    # print(pacf_value)
    
    # ------------------------------
    # statsmodels
    # ------------------------------
    get_acf_pacf(white_noise)
    get_acf_pacf(random_walk)

    # ------------------------------
    # plot
    # ------------------------------
    series_list = [white_noise, random_walk]
    series_names = ['white_noise', 'random_walk']
    plot_acf_pacf(series_list, series_names)

if __name__ == "__main__":
    main()
