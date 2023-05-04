# -*- coding: utf-8 -*-


# ***************************************************
# * File        : average_smoothing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-10
# * Version     : 0.1.111023
# * Description : Average Moving Smoothing(移动平均)
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np


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

