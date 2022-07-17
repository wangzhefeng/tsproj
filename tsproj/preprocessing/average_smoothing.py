"""
Average Smoothing 时间序列平均光滑
"""


import numpy as np


def average_smoothing(signal, kernel_size = 3, stride = 1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start) * np.mean(signal[start:end]))
    
    return np.array(sample)



if __name__ == "__main__":
    pass