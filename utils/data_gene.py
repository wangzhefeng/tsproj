# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_gene.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-18
# * Version     : 0.1.121803
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import akshare as ak
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.linear_model import LinearRegression

font_name = ["Arial Unicode MS"]
mpl.rcParams["font.sans-serif"] = font_name
mpl.rcParams["axes.unicode_minus"] = False


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def white_noise(timesteps: int, is_print: bool = False, is_plot: bool = False):
    """
    生成白噪声模拟数据数据
    """
    white_noise = np.random.standard_normal(size = timesteps)

    if is_print:
        print(white_noise)
    
    if is_plot:
        plt.figure(figsize = (12, 6))
        plt.plot(white_noise)
        plt.show()
    
    return white_noise


def non_white_noise(is_print: bool = False, is_plot: bool = False):
    """
    我国06年以来的季度GDP数据季节差分后, 就可以认为是一个平稳的时间序列

    Args:
        is_print (bool, optional): _description_. Defaults to False.
        is_plot (bool, optional): _description_. Defaults to False.
    """
    # data
    df = ak.macro_china_gdp()
    df = df.set_index("季度")
    df.index = pd.to_datetime(df.index)
    print(df.head())
    print(df.shape)
    # 原始数据
    gdp = df["国内生产总值-绝对值"][::-1].astype("float")
    print(gdp)
    # 差分
    gdp_diff = gdp.diff(4)
    print(gdp_diff)

    if is_plot:
        plt.figure(figsize = (12, 6))
        gdp_diff.plot()
        plt.show()

def white_noise_check(white_noise_series):
    """
    白噪声序列自相关性检验

    Args:
        white_noise_series (_type_): _description_
    """
    # 绘制白噪声序列图及其 ACF 图和 PACF 图
    fig = plt.figure(figsize = (20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    # 白噪声序列图
    ax1.plot(white_noise_series)
    ax1.set_title("white noise")
    # 白噪声序列 ACF 图
    plot_acf(white_noise_series, ax = ax2)
    # 白噪声序列 PACF 图
    plot_pacf(white_noise_series, ax = ax3)
    plt.show()


def gen_randomwalk(timesteps, noise, is_print: bool = False, is_plot: bool = False):
    """
    随机游走

    Args:
        timesteps (_type_): _description_
        noise (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = np.random.normal(0, noise, (timesteps,))
    series = y.cumsum()
    if is_print:
        print(series)

    if is_plot:
        plt.figure(figsize = (12, 6))
        plt.plot(series)
        plt.show()

    return series


def randomwalk_normal(timesteps, is_print: bool = False, is_plot: bool = False):
    """
    普通随机游走
    """
    y = np.random.standard_normal(size = timesteps)
    series = np.cumsum(y)
    if is_print:
        print(series)

    if is_plot:
        plt.figure(figsize = (12, 6))
        plt.plot(series)
        plt.show()

    return series


def randomwalk_drift(timesteps: int, drift: float,  is_print: bool = False, is_plot: bool = False):
    """
    带漂移项的随机游走
    """
    y = np.random.standard_normal(size = timesteps)
    y_cunsum = np.cumsum(drift + y)
    series = np.cumsum(drift * np.ones(len(y_cunsum)))
    if is_print:
        print(series)

    if is_plot:
        plt.figure(figsize = (12, 6))
        plt.plot(series)
        plt.show()
    
    return series


def gen_sinusoidal(timesteps, amp, freq, noise):
    """
    正弦曲线
    """
    X = np.arange(timesteps)
    e = np.random.normal(0, noise, (timesteps,))
    y = amp * np.sin(X * (2 * np.pi / freq)) + e

    return y


def gen_ts(timesteps, amp, freq, noise, random_state = 0):
    """
    _summary_
    """
    np.random.seed(random_state)
    
    if isinstance(freq, (int, float)):
        seas = gen_sinusoidal(timesteps = timesteps, amp = amp, freq = freq, noise = noise)
    elif np.iterable(freq) and not isinstance(freq, str):
        seas = np.zeros(timesteps)
        for f in freq:
            if isinstance(f, (int,float)):
                seas += gen_sinusoidal(timesteps = timesteps, amp = amp, freq = f, noise = noise)
            else:
                raise ValueError("freq not understood.")
    else:
        raise ValueError("freq not understood.")
    
    rw = gen_randomwalk(timesteps = timesteps, noise = 1)
    X = np.linspace(0, 10, timesteps).reshape(-1, 1)
    X = np.power(X, [1, 2])
    trend = LinearRegression().fit(X, rw).predict(X)
    
    return seas + trend


def non_stationarity(is_print: bool = False, is_plot: bool = False):
    """
    非平稳序列示例

    Args:
        is_print (bool, optional): _description_. Defaults to False.
        is_plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    df = ak.stock_zh_a_hist(
        symbol = "603777",
        start_date = "20190101",
        end_date = "20210616",
    )
    df = df.set_index("日期")
    df.index = pd.to_datetime(df.index)

    close = df["收盘"].astype(float)
    close = close[::-1]
    if is_print:
        print(close)

    if is_plot:
        plt.figure(figsize = (12, 6))
        plt.plot(close)
        plt.show()
    
    return close




# 测试代码 main 函数
def main():
    # white_noise_series = white_noise(size = 1000)
    # white_noise_check(white_noise_series)
    non_stationarity(is_plot = True)
    

if __name__ == "__main__":
    main()

