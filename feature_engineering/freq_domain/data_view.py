# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_view.py
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

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": [16, 10],
    "font.size": 18,
})

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def time_domain_series_view(x, y):
    """
    plot the time domain series(signal)

    Args:
        x (_type_): _description_
        y (_type_): _description_
    """
    plt.figure(figsize = (15, 5))
    plt.plot(x, y)
    plt.xlabel("Time(s)", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title("Time Domain Series")
    # plt.xlim()
    # plt.ylim()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.show()


def fourier_transform_signal_view(freq, y_abs):
    """
    plot the magnitude of the Fourier Transform

    Args:
        freq (_type_): _description_
        y_abs (_type_): _description_
    """
    plt.figure(figsize=(15, 5))
    plt.plot(freq, y_abs)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform of the Signal")
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.xticks([3.2, 1, 2])
    # plt.yticks() 
    plt.grid(True, alpha=0.5)
    plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
