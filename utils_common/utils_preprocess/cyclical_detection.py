# -*- coding: utf-8 -*-


# ***************************************************
# * File        : timeseries_fft.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-13
# * Version     : 0.1.021320
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
from itertools import cycle
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import logging
from typing import List, Dict
# 傅里叶变换
from scipy.fftpack import fft, fftfreq
# 自相关系数
from statsmodels.tsa.stattools import acf



def get_cycle_fft(data):
    """
    傅里叶变换估计周期

    Args:
        data ([type]): [description]
    """
    # 傅里叶变换
    fft_series = fft(data["value"].values)
    # 傅里叶系数长度的平方
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)

    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    powers = power[pos_mask]

    # 傅里叶变换得到的傅里叶系数长度的平方作为纵坐标, 相应索引与序列总长度的比值(频率)为横坐标,
    # 取凸起的三个点, 对应的周期为横坐标的倒数, 都可以看作候选周期
    top_k_seasons = 3
    top_k_idxs = np.argpartition(power, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)

    logging.error(f"top_{top_k_seasons}_power: {top_k_power}")
    logging.error(f"fft_periods: {fft_periods}")
    
    return fft_periods


def get_cycle_acf(data, fft_periods: List[int], expected_lags):
    """
    计算自相关系数

    Args:
        data ([type]): [description]
        fft_periods ([type]): [description]
        expected_lags (type): [description] np.array([
            timedelta(hours = 12) / timedelta(minutes = 5),  # 12hour
            timedelta(days = 1) / timedelta(minutes = 5),  # 1day
            timedelta(days = 7) / timedelta(minutes = 5)  # 7day
        ]).astype(int)
    """
    lags = []
    acf_scores = []
    # 计算傅里叶变换中得到的候选周期的自相关系数
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(data["value"].values, nlags = lag)[-1]
        logging.error(f"lag: {lag} fft acf: {acf_score}")
        lags.append(lag)
        acf_scores.append(acf_score)
    # 测试几个预设周期
    for lag in expected_lags:
        acf_score = acf(data["value"].values, nlags = lag, fft = False)[-1]
        logging.error(f"lag: {lag} expected acf: {acf_score}")
        lags.append(lag)
        acf_scores.append(acf_score)
    cycle = lags[np.argmax(acf_scores)]
    
    return cycle




# 测试代码 main 函数
def main():
    data = pd.read_csv("")
    fft_periods = get_cycle_fft(data)
    cycle = get_cycle_acf(data, fft_periods)


if __name__ == "__main__":
    main()

