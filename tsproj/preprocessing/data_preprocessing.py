# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_preprocessing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 序列数据预处理类函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def SeriesDataNormalize(data):
    """
    数据序列归一化函数, 受异常值影响
    Parameters: 
        data: np.array (n, m)
    Returns:
        scaler: 归一化对象
        normalized: 归一化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    normalized = scaler.transform(data)
    return scaler, normalized


def SeriesDataStandardScaler(data):
    """
    数据序列标准化函数, 不受异常值影响
    Parameters: 
        data: np.array (n, m)
    Returns:
        scaler: 标准化对象
        normalized: 标准化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(data)
    normalized = scaler.transform(data)
    return scaler, normalized


def MovingAverageSmooth(data, WSZ=11):
    """
    滑窗均值平均
    Parameters: 
        data: list, np.appay
        WSZ: window size
    """
    out0 = np.convolve(data, np.ones(WSZ, dtype=int), 'valid') / WSZ
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
            res.append(np.median(data[i:i+WSZ]))
        tail = [res[-1] for _ in range(WSZ)]
        return res + tail
    elif direction == "right":
        for i in range(len(data)-1, WSZ-1, -1):
            res.insert(0, np.median(data[i-WSZ:i]))
        tail = [res[0] for _ in range(WSZ)]
        return tail + res
    elif direction == "circle":
        n_data = data + data[:WSZ][::-1]
        for i in range(len(n_data) - WSZ):
            res.append(np.median(n_data[i:i+WSZ]))
        return res
    else:
        return None


def SeriesOutlierDetection(data, max_samples="auto", contamination="auto", threshold=None):
    """
    基于isolation forest算法的单特征时序数据异常值检测
    Parameters: 
        max_samples, contamination: 参考sklearn文档
        threshold: 异常值分数阈值
    Returns: 
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = IsolationForest(max_samples=max_samples, contamination=contamination)
    if threshold is None:
        return clf.fit_predict(s_data)
    else:
        label_res = []
        scores = clf.score_samples(s_data)
        for score_i in scores:
            if score_i < threshold:
                label_res.append(-1)
            else:
                label_res.append(1)
        return label_res




def main():
    pass


if __name__ == "__main__":
    main()

