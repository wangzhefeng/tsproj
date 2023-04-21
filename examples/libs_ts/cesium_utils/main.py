# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-10-29
# * Version     : 0.1.102923
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from cesium import datasets


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
PYTHONHTTPSVERIFY = 0


eeg = datasets.fetch_andrzejak()
# print(eeg)
print(eeg.keys())










# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

