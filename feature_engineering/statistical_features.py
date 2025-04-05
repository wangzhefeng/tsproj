# -*- coding: utf-8 -*-

# ***************************************************
# * File        : statistical_features.py
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def extract_statistical_features(y):
     def calculate_entropy(y):
         # Ensure y is positive and normalized
         y = np.abs(y)
         y_sum = np.sum(y)
 
         # Avoid division by zero
         if y_sum == 0:
             return 0
 
         # Normalize the signal
         p = y / y_sum
 
         # Calculate entropy
         entropy_value = -np.sum(p * np.log2(p + 1e-12))  # Add a small value to avoid log(0)
 
         return entropy_value
     # Remove the mean to center the data
     y_centered = y - np.mean(y)
     y = y+np.max(np.abs(y))*10**-4
 
     # Statistical features
     mean_value = np.mean(y)
     variance_value = np.var(y)
     skewness_value = skew(y)
     kurtosis_value = kurtosis(y)
     autocorrelation_value = np.correlate(y_centered, y_centered, mode='full')[len(y) - 1] / len(y)
     quantiles = np.percentile(y, [25, 50, 75])
     entropy_value = calculate_entropy(y)  # Add a small value to avoid log(0)
 
     # Create a dictionary of features
     statistical_dict = {
         'Mean': mean_value,
         'Variance': variance_value, 
         'Skewness': skewness_value, 
         'Kurtosis': kurtosis_value, 
         'Autocorrelation': autocorrelation_value, 
         'Quantile_25': quantiles[0], 
         'Quantile_50': quantiles[1], 
         'Quantile_75': quantiles[2], 
         'Entropy': entropy_value
    }
 
     # Convert to DataFrame for easy visualization and manipulation
     statistical_data = pd.DataFrame([statistical_dict])
 
     return statistical_dict, statistical_data


def extract_peaks_and_valleys(y, N=10):
    """
    https://docs.scipy.org/doc/scipy/reference/signal.html#peak-finding

    Args:
        y (_type_): _description_
        N (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # Find peaks and valleys
    peaks, _ = find_peaks(y)
    valleys, _ = find_peaks(-y)

    # Combine peaks and valleys
    all_extrema = np.concatenate((peaks, valleys))
    all_values = np.concatenate((y[peaks], -y[valleys]))

    # Sort by absolute amplitude (largest first)
    sorted_indices = np.argsort(-np.abs(all_values))
    sorted_extrema = all_extrema[sorted_indices]
    sorted_values = all_values[sorted_indices]

    # Select the top N extrema
    top_extrema = sorted_extrema[:N]
    top_values = sorted_values[:N]

    # Pad with zeros if fewer than N extrema are found
    if len(top_extrema) < N:
        padding = 10 - len(top_extrema)
        top_extrema = np.pad(top_extrema, (0, padding), 'constant', constant_values=0)
        top_values = np.pad(top_values, (0, padding), 'constant', constant_values=0)

    # Prepare the features
    features = []
    for i in range(N):
        features.append(top_values[i])
        features.append(top_extrema[i])

    # Create a dictionary of features
    feature_dict = {f'peak_{i+1}': features[2*i] for i in range(N)}
    feature_dict.update({f'loc_{i+1}': features[2*i+1] for i in range(N)})

    return feature_dict, pd.DataFrame([feature_dict])




# 测试代码 main 函数
def main():
    from utils.log_util import logger

    # ------------------------------
    # data
    # ------------------------------    
    # data
    from feature_engineering.freq_domain.data_factory import generate_series_with_three_components
    x, y = generate_series_with_three_components()

    # time domain data view
    from feature_engineering.freq_domain.data_view import time_domain_series_view
    time_domain_series_view(x, y)

    # ------------------------------
    # statistical features
    # ------------------------------
    statistical_dict, statistical_data = extract_statistical_features(y)
    logger.info(statistical_data)

    # ------------------------------
    # find peaks
    # ------------------------------
    # data
    from scipy.datasets import electrocardiogram
    x = electrocardiogram()[0:2000]
    logger.info(f"x: {x}, \nx type{type(x)}")
    
    # find peaks
    peak_height_threshold = -0.5
    peaks, properties = find_peaks(x, height = peak_height_threshold)

    # data view
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(x)), x, label="series")
    plt.plot(peaks, x[peaks], "x", label="peaks")
    plt.axhline(y=peak_height_threshold, ls="--", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title('Series')
    plt.grid(True)
    plt.show();
    # ------------------------------
    # peaks and valleys features
    # ------------------------------
    features = extract_peaks_and_valleys(y, N=10)
    logger.info(features[1])

if __name__ == "__main__":
    main()
