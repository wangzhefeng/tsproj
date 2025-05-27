# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wavelet.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040516
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

import numpy as np
import pandas as pd
import pywt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def extract_wavelet_features(y, wavelet='db4', level=3, num_features=5):
    # Remove the mean
    y = y - np.mean(y)

    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # Flatten the list of coefficients into a single array
    coeffs_flat = np.hstack(coeffs)

    # Get the absolute values of the coefficients
    coeffs_abs = np.abs(coeffs_flat)

    # Find the indices of the largest coefficients
    largest_coeff_indices = np.flip(np.argsort(coeffs_abs))[0:num_features]

    # Extract the largest coefficients as features
    top_coeffs = coeffs_flat[largest_coeff_indices]

    # Generate feature names for the wavelet features
    feature_keys = ['Wavelet Coeff ' + str(i+1) for i in range(num_features)]

    # Create a dictionary for the features
    wavelet_dict = {
        feature_keys[i]:top_coeffs[i] 
        for i in range(num_features)
    }

    # Create a DataFrame for the features
    wavelet_data = pd.DataFrame(top_coeffs).T
    wavelet_data.columns = feature_keys

    return wavelet_dict, wavelet_data




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    # data
    from feature_engineering.freq_domain.data_factory import generate_series_with_three_components
    x, y = generate_series_with_three_components()

    # time domain data view
    from feature_engineering.freq_domain.data_view import time_domain_series_view
    time_domain_series_view(x, y)

    # Example usage:
    wavelet_dict, wavelet_data = extract_wavelet_features(y)
    logger.info(f"wavelet_data: \n{wavelet_data}")

if __name__ == "__main__":
    main()
