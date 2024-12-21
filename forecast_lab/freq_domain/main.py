# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-04
# * Version     : 0.1.110421
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# python libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": [16, 10],
    "font.size": 18,
})

# ------------------------------
# data
# ------------------------------
# 时间序列
t = np.arange(start = 0, stop = 1, step = 0.001)

# 正弦波序列
freq_50_series = np.sin(2 * np.pi * 50 * t)
freq_120_series = np.sin(2 * np.pi * 120 * t)
# 正弦波序列组合
f_clean = freq_50_series + freq_120_series
# 噪声数据
noise = 2.5 * np.random.randn(len(t))
# 噪声污染序列
f_noise = f_clean + noise



# 测试代码 main 函数
def main():
    import numpy as np
    
    def DFT_slow(x):
        """
        Compute the discrete Fourier Transform of the 1D array x
        """
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        X = np.dot(M, x)

        return X

    x = np.random.random(1024)
    print(np.allclose(DFT_slow(x), np.fft.fft(x)))

    x_fft_slow = DFT_slow(x)
    x_fft_np = np.fft.fft(x)
    print(x)
    print(x_fft_slow)
    print(x_fft_np)

if __name__ == "__main__":
    main()
