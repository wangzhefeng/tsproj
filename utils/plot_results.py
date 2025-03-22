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


def plot_results(preds, trues, title: str = "result"):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(trues, label = 'True Data')
    plt.plot(preds, label = 'Prediction')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(f'images/{title}_results.png')


def plot_results_multiple(preds, trues, preds_len: int, title: str = "results_multiple"):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(trues, label = 'True Data')
    for i, data in enumerate(preds):
        padding = [None for p in range(i * preds_len)]
        plt.plot(padding + data, label = 'Prediction')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(f'images/{title}_results_multiple.png')




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
