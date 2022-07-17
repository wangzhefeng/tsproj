"""
Wavelet denoising 小波去燥
"""


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



if __name__ == "__main__":
    pass