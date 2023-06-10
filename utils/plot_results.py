# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052220
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

import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_results(predicted_data, 
                 true_data, 
                 title: str = "result", 
                 is_show: bool = True, 
                 is_save: bool = True):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label = 'True Data')
    plt.plot(predicted_data, label = 'Prediction')
    plt.legend()
    plt.title(title)
    # 图片展示
    if is_show:
        plt.show()
    # 图片保存
    if is_save:
        plt.savefig(f'images/{title}_results.png')


def plot_results_multiple(predicted_data, 
                          true_data, 
                          prediction_len: int, 
                          title: str = "results_multiple", 
                          is_show: bool = True, 
                          is_save: bool = True):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label = 'True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label = 'Prediction')
    plt.legend()
    plt.title(title)
    # 图片展示
    if is_show:
        plt.show()
    # 图片保存
    if is_save:
        plt.savefig(f'images/{title}_results_multiple.png')


def plot_train_results(pred, true):
    plot_size = 200
    plt.figure(figsize = (12, 8))
    plt.plot(pred, "b")
    plt.plot(true, "r")
    plt.legend()
    plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
