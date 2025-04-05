# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040517
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

import numpy as np

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def generate_series_with_noise():
    # time steps
    data_step = 0.001
    t = np.arange(start = 0, stop = 1, step = data_step)

    # 正弦波序列
    freq_50_series = np.sin(2 * np.pi * 50 * t)
    freq_120_series = np.sin(2 * np.pi * 120 * t)
    # 正弦波序列组合
    f_clean = freq_50_series + freq_120_series

    # 噪声数据
    noise_series = 2.5 * np.random.randn(len(t))
    # 噪声污染序列
    f_noise = f_clean + noise_series
    
    return t, f_noise


def generate_series_with_three_components():
    """
    generate series with three components

    三个主要成分
        - 振幅=1.0, 频率=1.0
        - 振幅=0.4, 频率=2.0
        - 振幅=2.0, 频率=3.2
    """
    # time, range: [-8pi, 2pi], period = 10000
    x = np.linspace(-8 * np.pi, 8 * np.pi, 10000)
    # signal 
    y = np.sin(x) + 0.4 * np.cos(2 * x) + 2 * np.sin(3.2 * x)

    return x, y


def generate_sine_wave(freq = 2, sample_rate = 400000, duration = 5):
    """
    Generate a 2 hertz sine wave that lasts for 5 seconds

    Args:
        freq (_type_): frequency
        sample_rate (int, optional): Hertz. Defaults to 400000.
        duration (int, optional): Seconds. Defaults to 5.

    Returns:
        _type_: _description_
    """
    # x axis
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    # freq
    frequencies = x * freq
    # y axis, 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)

    return x, y




# 测试代码 main 函数
def main():
    from feature_engineering.freq_domain.data_view import time_domain_series_view

    # 生成序列
    x, y = generate_series_with_noise()
    time_domain_series_view(x, y)

    # 生成序列
    x, y = generate_series_with_three_components()
    time_domain_series_view(x, y)

    # 生成正弦波
    x, y = generate_sine_wave()
    time_domain_series_view(x, y)

if __name__ == "__main__":
    main()
