# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_processing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 序列数据预处理类函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
import pysindy as ps
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def GetSeriesDerivative(x_data, y_data, dev_method="Spline"):
    """
    依据给定方法求y_data序列对x_data的导数
    Parameters:
        x_data: 变量x序列
        y_data: 因变量序列
        dev_method: 微分方法
    Returns:
        微分结果
    """
    diffs = {
        'FiniteDifference': ps.FiniteDifference(),
        'Finite Difference': ps.SINDyDerivative(kind='finite_difference', k=1),
        'SmoothedFiniteDifference': ps.SmoothedFiniteDifference(),
        'SavitzkyGolay': ps.SINDyDerivative(kind='savitzky_golay', left=0.5, right=0.5, order=3),
        'Spline': ps.SINDyDerivative(kind='spline', s=1e-2),
        'TrendFiltered': ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-2),
        'Spectral': ps.SINDyDerivative(kind='spectral')
    }
    if diffs.get(dev_method) is not None:
        return diffs[dev_method](y_data, x_data)
    return None


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


def IsolationForestOutlierDetection(data, max_samples="auto", contamination="auto", threshold=None):
    """
    基于isolation forest算法的单特征时序数据异常值检测
    Parameters: 
        data: series list
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


def OneClassSvmOutlierDetection(data, kernel="brf", gamma=0.1, nu=0.3):
    """
    基于OneClassSvm算法的单特征时序数据异常值检测
    Parameters: 
        data: series list
        kernel:
        threshold: 异常值分数阈值
    Returns: 
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
    error_data = np.asarray(data).reshape(-1, 1)

    # fit the model
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    res = clf.fit_predict(error_data)
    return res


def LofOutlierDetection(data, neighbor=50, dist_metric="l1", contamination="auto"):
    """
    基于LOF算法的单特征时序数据异常值检测
    Parameters: 
        data: series list
        neighbor: 近邻数
        dist_metric: 距离计算方法
        contamination: 异常值比例
    Returns: 
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = LOF(n_neighbors=neighbor, metric=dist_metric, contamination=contamination)
    res = clf.fit_predict(s_data)
    return res




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

