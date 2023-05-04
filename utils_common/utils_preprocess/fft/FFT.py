# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ft.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-30
# * Version     : 0.1.113000
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": [16, 10],
    "font.size": 18,
})

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class FFT:
    
    def __init__(self) -> None:
        pass
    
    def fft(self):
        pass

    def noise_remove(self):
        pass

    def seasonal_detection(self):
        pass




def unit_test(self):
    # ------------------------------
    # data
    # ------------------------------
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






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

