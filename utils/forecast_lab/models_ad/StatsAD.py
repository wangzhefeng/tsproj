# -*- coding: utf-8 -*-


# ***************************************************
# * File        : StatsAD.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-30
# * Version     : 0.1.033022
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import zscore


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def NSigma(series, num_sigma: int = 3):
    """
    N Sigma Anomaly Detection

    Data should be normally distributed. 
    Under the 3-sigma principle, an outlier 
    that exceeds n standard deviations can 
    be regarded as an outlier  

    Args:
        series (_type_): pd.DataFrame with only one column or pd.Series
        num_sigma (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: the index list of outlier data
    """
    mu, std = series.mean(), series.std()
    lower, upper = mu - num_sigma * std, mu + num_sigma * std

    outlier_series = series[(lower >= series) | (upper <= series)]
    normal_series = series[(lower < series) & (upper > series)]
    labels = list(outlier_series.dropna().index)
    series = outlier_series.dropna()

    return labels, series


def ZScore(series, threshold):
    """
    Z-Score 标准分数异常值检测

    Args:
        series (_type_): _description_
        threshold (_type_): 一般取值为：2.5, 3.0, 3.5

    Returns:
        _type_: _description_
    """
    z_score = (series - np.mean(series)) / np.std(series)
    outlier_series = series[z_score >= threshold]
    normal_series = series[z_score < threshold]
    labels = list(outlier_series.dropna().index)
    series = outlier_series.dropna()

    return labels, series


def ZScoreAD(df, features, threshold = 2.5):
    '''
    df: 表示要处理的 dataFrame
    features: 表示要处理的特征
    threshold: 表示判断是否是异常值的阈值
    '''
    for col in features:
        all_value = df[col].values.copy()
        ## 先剔除掉所有 nan 值
        indices = np.array(list(
            map(lambda x: not x, np.isnan(all_value))
        ))
        true_value = all_value[indices]
        z_value = zscore(true_value)
        all_value[indices] = z_value
        ## 得到 zscore 所在的列，以及是否是异常值的列
        df[col + "_zscore"] = all_value
        df[col + "_isanomaly"] = df[col + "_zscore"].apply(lambda x: int(abs(x) >= abs(threshold)))


def BoxPlot(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return lower, upper


def Grubbs():
    """
    Grubbs 检验
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm
    https://github.com/c-bata/outlier-utils
    """
    from outliers import smirnov_grubbs as grubbs
    print(grubbs.test([8, 9, 10, 1, 9], alpha = 0.05))
    print(grubbs.min_test_outliers([8, 9, 10, 1, 9], alpha = 0.05))
    print(grubbs.max_test_outliers([8, 9, 10, 1, 9], alpha = 0.05))
    print(grubbs.max_test_indices([8, 9, 10, 50, 9], alpha = 0.05))


def Threshold(df, lower_limit, upper_limit):
    """
    Data greater than the upper limit or less 
    than the lower limit is considered as an outlier.

    Args:
        df (_type_): pd.DataFrame/pd.Series with only one column
        lower_limit (_type_): lower limit of the df value
        upper_limit (_type_): upper limit of the df value

    Returns:
        _type_: the index list of outlier data
    """
    outlier_df = df[(df > upper_limit) | (df < lower_limit)]
    labels = list(outlier_df.dropna().index)
    
    return labels


def statistics(df, mean_min = None, mean_max = None, variance_max = None):
    """
    Statistics data(about 30 minutes) to judge if it's an anomaly.

    Args:
        - df: should be pd.DataFrame with only one column
        - mean_min: minimum mean value
        - mean_max: maximum mean value
        - variance_max: maximum variance value
    Return: 
        - if the data abnormal. True-abnormal, false-normal
    """
    if mean_min is not None:
        if df.mean() < mean_min:
            return True
    if mean_max is not None:
        if df.mean() > mean_max:
            return True
    if variance_max is not None:
        if df.var() > variance_max:
            return True
    
    return False


def KL(y1, y2):
    """
    Calculate the similarity of two sets datas' distribution.

    Args
        y1, y2: list
    Return: 
        KL value, the smaller, the closer.
    """
    return scipy.stats.entropy(y1, y2) 


def CorrelationAD(df, method = "pearson", threshold = 0.5):
    """
    根据事先离线分析出的两个变量直接的相关性大小, 设定阈值。
    一段时间内两个变量的相关性低于阈值, 认为存在异常

    Args:
        df (_type_): _description_
        method (str, optional): 对应着三种相关性计算方法: pearson, kendall, spearman. Defaults to "pearson".
            皮尔逊相关系数(pearson):连续性变量才可采用
            肯达相关系数(kendall):反映分类变量相关性的指标,适用于两个分类变量均为有序分类的情况
            斯皮尔曼相关系数(spearman):利用两变量的秩次大小作线性相关分析,对原始变量的分布不作要求,属于非参数统计方法,适用范围要广些
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: 两个变量相关性是否异常。异常为True, 正常为False
    """
    return df.corr(method = method).iloc[1][0] < threshold




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
