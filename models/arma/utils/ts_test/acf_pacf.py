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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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


def get_acf_pacf(series: pd.Series, nlags: int = 3):
    """
    ACF，PACF
    """
    lag_acf = sm.tsa.stattools.acf(series, nlags = nlags)
    lag_pacf = sm.tsa.stattools.pacf(series, nlags = nlags, method = "ols")

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




# 测试代码 main 函数
def main():
    """
    import akshare as ak
    # 白噪声
    white_noise = np.random.standard_normal(size = 1000)

    # 随机游走
    x = np.random.standard_normal(size = 1000)
    random_walk = np.cumsum(x)

    # GDP
    df = ak.macro_china_gdp()
    df = df.set_index('季度')
    df.index = pd.to_datetime(df.index)
    gdp = df['国内生产总值-绝对值'][::-1].astype('float')

    # GDP DIFF
    gdp_diff = gdp.diff(4).dropna() 
    
    # acf, pacf plot
    fig, ax = plt.subplots(4, 2)
    fig.subplots_adjust(hspace = 0.5)

    plot_acf(white_noise, ax = ax[0][0])
    ax[0][0].set_title('ACF(white_noise)')
    plot_pacf(white_noise, ax = ax[0][1])
    ax[0][1].set_title('PACF(white_noise)')

    plot_acf(random_walk, ax = ax[1][0])
    ax[1][0].set_title('ACF(random_walk)')
    plot_pacf(random_walk, ax = ax[1][1])
    ax[1][1].set_title('PACF(random_walk)')

    plot_acf(gdp, ax = ax[2][0])
    ax[2][0].set_title('ACF(gdp)')
    plot_pacf(gdp, ax = ax[2][1])
    ax[2][1].set_title('PACF(gdp)')

    plot_acf(gdp_diff, ax = ax[3][0])
    ax[3][0].set_title('ACF(gdp_diff)')
    plot_pacf(gdp_diff, ax = ax[3][1])
    ax[3][1].set_title('PACF(gdp_diff)')

    plt.show()
    """
    # vecvor
    # x = pm.c(1, 2, 3, 4, 5, 6, 7)
    # print(x)
    # acf_value = get_acf(x)
    # print(acf_value)

    # pacf_value = get_pacf(x)
    # print(pacf_value)
    
    # ------------------------------
    # 
    # ------------------------------
    x = pd.Series([1, 2, 3, 4, 5, 6, 7])
    get_acf_pacf(x)

if __name__ == "__main__":
    main()
