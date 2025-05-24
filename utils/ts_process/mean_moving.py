# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mean_moving.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-10
# * Version     : 1.0.091023
# * Description : Mean Moving Smoothing(移动平均)
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


def average_moving_smoothing(signal, kernel_size = 3, stride = 1):
    sample = []
    # 时序数据窗口开始结束索引
    start, end = 0, kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start) * np.mean(signal[start:end]))
    
    return np.array(sample)


def weight_average_moving_smoothing(signal, kernel_size = 3, stride = 1):
    pass





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
