# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ft.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-30
# * Version     : 0.1.113000
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def extract_fft_features(y, x=None, num_features = 5, max_frequency = 10):
    """
    extract fft features

    Args:
        y (_type_): _description_
        x (_type_, optional): _description_. Defaults to None.
        num_features (int, optional): _description_. Defaults to 5.
        max_frequency (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # Remove the mean
    y = y - np.mean(y)

    # Perform the Fourier Transform
    Y = np.fft.fft(y)

    # Calculate the frequency bins
    if x is None:
        x = np.linspace(0, len(y))
    frequencies = np.fft.fftfreq(len(x), d = (x[1] - x[0]) / (2 * np.pi))

    # Normalize the amplitude(振幅) of the FFT
    Y_abs = 2 * np.abs(Y) / len(x)

    # Zero out very small values to remove noise
    Y_abs[Y_abs < 1e-6] = 0

    relevant_frequencies = np.where((frequencies > 0) & (frequencies < max_frequency))
    Y_phase = np.angle(Y)[relevant_frequencies]
    frequencies = frequencies[relevant_frequencies]
    Y_abs = Y_abs[relevant_frequencies]
    # TODO
    # largest amplitudes
    largest_amplitudes = np.flip(np.argsort(Y_abs))[0:num_features]
    top_5_amplitude = Y_abs[largest_amplitudes]
    top_5_frequencies = frequencies[largest_amplitudes]
    top_5_phases = Y_phase[largest_amplitudes]

    # Create a dictionary for the features
    fft_features = top_5_amplitude.tolist() + top_5_frequencies.tolist() + top_5_phases.tolist()
    # amplitude 振幅
    amp_keys = ['Amplitude '+str(i) for i in range(1, num_features+1)]
    # frequency, 频率
    freq_keys = ['Frequency '+str(i) for i in range(1, num_features+1)]
    # phase, 相位
    phase_keys = ['Phase '+str(i) for i in range(1, num_features+1)]
    
    fft_keys = amp_keys + freq_keys + phase_keys
    fft_dict = {
        fft_keys[i]:fft_features[i] 
        for i in range(len(fft_keys))
    }
    fft_data = pd.DataFrame(fft_features).T
    fft_data.columns = fft_keys
    
    return fft_dict, fft_data




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    '''
    # ------------------------------
    # Discrete Fourier Transform
    # ------------------------------
    def DFT_slow(x):
        """
        Compute the discrete Fourier Transform of the 1D array x
        """
        x = np.asarray(x, dtype=float)
        
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        
        # M @ x
        X = np.dot(M, x)

        return X
    # series
    x = np.random.random(1024)
    print(np.allclose(DFT_slow(x), np.fft.fft(x)))

    # two method of fft
    x_fft_slow = DFT_slow(x)
    x_fft_np = np.fft.fft(x)
    print(x)
    print(x_fft_slow)
    print(x_fft_np)

    data_view(range(len(x)), x)
    data_view(range(len(x_fft_slow)), x_fft_slow)
    data_view(range(len(x_fft_np)), x_fft_np)
    '''
    # data
    from feature_engineering.freq_domain.data_factory import generate_series_with_three_components
    x, y = generate_series_with_three_components()

    # time domain data view
    from feature_engineering.freq_domain.data_view import time_domain_series_view
    time_domain_series_view(x, y)

    # fft
    fft_dict, fft_data = extract_fft_features(y, x)
    logger.info(f"fft_data: \n{fft_data}")

if __name__ == "__main__":
    main()
