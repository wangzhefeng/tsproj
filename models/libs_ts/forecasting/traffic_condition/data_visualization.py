# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_visiual.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-27
# * Version     : 0.1.032718
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt



def speeds_array_ts_plot(speeds_array):
    plt.figure(figsize = (18, 6))
    plt.plot(speeds_array[:, [0, -1]])
    plt.legend(["route_0", "route_25"])
    plt.show()


def speeds_array_heatmap(speeds_array):
    plt.figure(figsize = (8, 8))
    plt.matshow(np.corrcoef(speeds_array.T), 0)
    plt.xlabel("rode number")
    plt.ylabel("rode number")
    plt.show()





# 测试代码 main 函数
def main():
    from utils_data.data_loader import data_loader

    route_distances, speeds_array = data_loader()

    speeds_array_ts_plot(speeds_array)
    speeds_array_heatmap(speeds_array)


if __name__ == "__main__":
    main()

