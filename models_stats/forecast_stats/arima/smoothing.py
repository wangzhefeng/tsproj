# -*- coding: utf-8 -*-

# ***************************************************
# * File        : smoothing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Union

from utils.log_util import logger
import numpy as np
from tsmoothie.smoother import KalmanSmoother, LowessSmoother

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def OnOffSmooth(data, threshold=0.5, WSZ=5, condition="and"):
    """
    开关类时序数据处理, 输出结果为0, 1组成的array
    Parameters:
        data: 序列数据, np.array
        threshold: 阈值
        WSZ, window size 窗口尺寸,奇数
        condition, and or
    """
    assert len(data) >= WSZ
    h_size = int(WSZ) // 2
    res = []
    for i in range(len(data)):
        l_bound, r_bound = i - h_size, i + h_size + 1
        if i - h_size < 0:
            l_bound = 0
        if r_bound >= len(data):
            r_bound = len(data)
        sec_i = np.asarray(data[l_bound:r_bound]) > threshold
        if condition == "and":
            if np.alltrue(sec_i):
                res.append(1)
            else:
                res.append(0)
        else:
            if any(data):
                res.append(1)
            else:
                res.append(0)
    return np.asarray(res)


def MovingAverageSmooth(data, WSZ=11):
    """
    滑窗均值平均
    Parameters, 
        data: list, np.array
        WSZ: window size
    """
    if data is None:
        return data

    if len(data) < WSZ:
        return data
    out0 = np.convolve(data, np.ones(WSZ, dtype = int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(data[:WSZ - 1])[::2] / r
    stop = (np.cumsum(data[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def MovingMedianSmooth(data, WSZ=11, direction="left"):
    """
    中值滤波, 
    Parameters, 
        data: list, np.appay
        WSZ: window size
        direction, 
            left: 从左到右, 用最后一个数据补齐右端
            right, 从右到左, 用最后一个数据补齐左端
            circle: 循环补齐
    """
    if len(data) < WSZ:
        return data

    res = []
    if direction == "left":
        for i in range(len(data) - WSZ):
            res.append(np.median(data[i:i + WSZ]))
        tail = [res[-1] for _ in range(WSZ)]
        return res + tail
    elif direction == "right":
        for i in range(len(data) - 1, WSZ - 1, -1):
            res.insert(0, np.median(data[i - WSZ:i]))
        tail = [res[0] for _ in range(WSZ)]
        return tail + res
    elif direction == "circle":
        n_data = data + data[:WSZ][::-1]
        for i in range(len(n_data) - WSZ):
            res.append(np.median(n_data[i:i + WSZ]))
        return res
    else:
        return None


def LowessSmooth(data_array, frac=0.01, iters=1, batch_size=None):
    """
    局部加权平均, 光滑时序数据, 不可缺失
    Parameters:
        data_array: np.array, [var_num, time_steps]
        frac: The smoothing span. A larger value of smooth_fraction
        will result in a smoother curve.
        iters: Between 1 and 6. The number of residual-based re_weightings to perform.
    Returns, 
        返回滤波后的序列数据
    Link, https://github.com/butayama/tsmoothie
    """
    iters = min(6, max(iters, 1))
    batch_size = min(data_array.shape[0], batch_size)
    smoother = LowessSmoother(smooth_fraction=frac, iterations=iters, batch_size=batch_size)
    smoother.smooth(data_array)
    low, up = smoother.get_intervals('prediction_interval')
    return smoother.smooth_data, low, up


def KalmanSmooth(data_array, component="level_trend", noise=dict(level=0.1, trend = 0.1)):
    """
    kalman smooth, 光滑含缺失的时序数据, 适合噪声大的数据, 缺失数据为np.nan
    Parameters:
        data_array: np.array, [var_num, time_steps]
        component: Specify the patterns and the dynamics present in our series
        noise: Specify in a dictionary the noise (in float term) of each single
            component provided in the 'component' argument
    Returns, 
        返回滤波后的序列数据
    Link, https://github.com/butayama/tsmoothie
    """
    m, n = data_array.shape
    if n < 2:
        return data_array, None, None
    smoother = KalmanSmoother(component=component, component_noise=noise)
    smoother.smooth(data_array)
    low, up = smoother.get_intervals('kalman_interval')
    return smoother.smooth_data, low, up


def OnOffSmooth(data, threshold=0.5, WSZ=5, condition="and"):
    """
    开关类时序数据处理, 输出结果为0, 1组成的array
    Parameters:
        data: 序列数据, np.array
        threshold: 阈值
        WSZ: window size 窗口尺寸,奇数
        condition: and or
    """
    assert len(data) >= WSZ
    h_size = int(WSZ) // 2
    res = []
    for i in range(len(data)):
        l_bound, r_bound = i - h_size, i + h_size + 1
        if i - h_size < 0:
            l_bound = 0
        if r_bound >= len(data):
            r_bound = len(data)
        sec_i = np.asarray(data[l_bound:r_bound]) > threshold
        if condition == "and":
            if np.alltrue(sec_i):
                res.append(1)
            else:
                res.append(0)
        else:
            if any(data):
                res.append(1)
            else:
                res.append(0)
    return np.asarray(res)


def MovingAverageSmooth(data: Union[list, np.ndarray], WSZ = 11):
    """
    滑窗均值平均

    Args:
        data (Union[list, np.ndarray]): _description_
        WSZ (int, optional): window size. Defaults to 11.

    Returns:
        _type_: _description_
    """
    if data is None:
        return data

    if len(data) < WSZ:
        return data
    out0 = np.convolve(data, np.ones(WSZ, dtype = int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(data[:WSZ - 1])[::2] / r
    stop = (np.cumsum(data[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def MovingMedianSmooth(data, WSZ=11, direction="left"):
    """
    中值滤波: 
    Parameters: 
        data: list, np.appay
        WSZ: window size
        direction: 
            left: 从左到右, 用最后一个数据补齐右端
            right: 从右到左, 用最后一个数据补齐左端
            circle: 循环补齐
    """
    if len(data) < WSZ:
        return data

    res = []
    if direction == "left":
        for i in range(len(data) - WSZ):
            res.append(np.median(data[i:i + WSZ]))
        tail = [res[-1] for _ in range(WSZ)]
        return res + tail
    elif direction == "right":
        for i in range(len(data) - 1, WSZ - 1, -1):
            res.insert(0, np.median(data[i - WSZ:i]))
        tail = [res[0] for _ in range(WSZ)]
        return tail + res
    elif direction == "circle":
        n_data = data + data[:WSZ][::-1]
        for i in range(len(n_data) - WSZ):
            res.append(np.median(n_data[i:i + WSZ]))
        return res
    else:
        return None


def LowessSmooth(data_array, frac=0.01, iters=1, batch_size=None):
    """
    局部加权平均, 光滑时序数据, 不可缺失
    Parameters:
        data_array: np.array, [var_num, time_steps]
        frac: The smoothing span. A larger value of smooth_fraction
        will result in a smoother curve.
        iters: Between 1 and 6. The number of residual-based re_weightings to perform.
    Returns: 
        返回滤波后的序列数据
    Link: https://github.com/butayama/tsmoothie
    """
    iters = min(6, max(iters, 1))
    batch_size = min(data_array.shape[0], batch_size)
    smoother = LowessSmoother(smooth_fraction=frac, iterations=iters, batch_size=batch_size)
    smoother.smooth(data_array)
    low, up = smoother.get_intervals('prediction_interval')
    return smoother.smooth_data, low, up


def KalmanSmooth(data_array, component="level_trend", noise=dict(level=0.1, trend = 0.1)):
    """
    kalman smooth, 光滑含缺失的时序数据, 适合噪声大的数据, 缺失数据为np.nan
    Parameters:
        data_array: np.array, [var_num, time_steps]
        component: Specify the patterns and the dynamics present in our series
        noise: Specify in a dictionary the noise (in float term) of each single
            component provided in the 'component' argument
    Returns: 
        返回滤波后的序列数据
    Link: https://github.com/butayama/tsmoothie
    """
    m, n = data_array.shape
    if n < 2:
        return data_array, None, None
    smoother = KalmanSmoother(component=component, component_noise=noise)
    smoother.smooth(data_array)
    low, up = smoother.get_intervals('kalman_interval')
    return smoother.smooth_data, low, up




def main():
    from tsmoothie.utils_func import sim_randomwalk
    np.random.seed(123)
    data = sim_randomwalk(n_series=2, timesteps=20, process_noise=10, measure_noise=30)
    logger.info(data.shape)
    for i in range(2):
        nan_cols = np.random.randint(20, size = 10)
        data[i, nan_cols] = np.nan

    # MovingAverageSmooth(data[0, :], WSZ=11)
    smooth_data, _, _ = KalmanSmooth(data)
    logger.info(smooth_data.shape)


if __name__ == "__main__":
    main()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
