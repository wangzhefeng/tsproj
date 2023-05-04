# -*- coding: utf-8 -*-


# ***************************************************
# * File        : watelet_denoising.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-02
# * Version     : 0.1.110200
# * Description : Wavelet denoising 小波去燥
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
import pywt


def maddest(d, axis = None):
    """
    MAD(mean absolute deviation)
    """
    mad = np.mean(np.absolute(d - np.mean(d, axis)), axis)
    return mad


def denoise_signal(series, wavelet = "db4", level = 1):
    """
    Wavelet denoising 小波去燥
    """
    # 小波系数
    coeff = pywt.wavedec(series, wavelet, mode = "per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    # 阈值
    uthresh = sigma * np.sqrt(2 * np.log(len(series)))
    coeff[1:] = (pywt.threshold(i, value = uthresh, mode = "hard") for i in coeff[1:])
    denoised_series = pywt.waverec(coeff, wavelet, mode = "per")
    return denoised_series










# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

