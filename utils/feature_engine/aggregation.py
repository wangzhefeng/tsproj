# -*- coding: utf-8 -*-


# ***************************************************
# * File        : Aggregation.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-18
# * Version     : 0.1.071822
# * Description : 聚合函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
from ctypes import Union
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None, 'display.max_rows', None)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def aggregation(df: pd.DataFrame, methods = None, group_col: str = None):
    """
    数据聚合

    Pandas 内置的 13 种内置聚合函数:
        pd.mean()  # 分组均值
        pd.sum()  # 分组求和
        pd.size()  # 分组个数
        pd.count()  # 分组大小
        pd.std()  # 分组标准差
        pd.var()  # 分组方差
        pd.sem()  # 均值误差
        pd.describe()  # 分组描述
        pd.first()  # 分组第一个元素
        pd.last()  # 分组最后一个元素
        pd.nth()  # 分组第 N 个元素
        pd.min()  # 分组最小值
        pd.max()  # 分组最大值

    :param df: _description_
    :type df: pd.DataFrame
    :param methods: _description_, defaults to None
    :type methods: str, optional
    :param group_col: _description_, defaults to None
    :type group_col: str, optional
    :return: _description_
    :rtype: _type_
    """
    if methods is None:
        logging.info(f"{LOGGING_LABEL} methods is None, please input correct 'methods' parameter.")
        # TODO raise Exception
        return None
    
    if group_col is None:
        logging.info(f"{LOGGING_LABEL} group column is None, please input correct 'group_col' parameter.")
        # TODO raise Exception
        return None
    
    # 聚合
    if isinstance(methods, str):
        if methods == "mean":
            return df.groupby(group_col).mean()
        elif methods == "sum":
            return df.groupby(group_col).sum()
        elif methods == "size":
            return df.groupby(group_col).size()
        elif methods == "count":
            return df.groupby(group_col).count()
        elif methods == "std":
            return df.groupby(group_col).std()
        elif methods == "var":
            return df.groupby(group_col).var()
        elif methods == "sem":
            return df.groupby(group_col).sem()
        elif methods == "describe":
            return df.groupby(group_col).describe()
        elif methods == "first":
            return df.groupby(group_col).first()
        elif methods == "last":
            return df.groupby(group_col).last()
        elif methods == "nth":
            return df.groupby(group_col).nth()
        elif methods == "min":
            return df.groupby(group_col).min()
        elif methods == "max":
            return df.groupby(group_col).max()
    elif isinstance(methods, list):
        """
        df.groupby(group_col).agg([np.mean, np.std])
        """
        return df.groupby(group_col).agg(methods)
    elif isinstance(methods, dict):
        """
        df.groupby(group_col).agg({
            col1: ["mean", "median",...], 
            col2: ["mean", "std",...], 
        })      
        """
        return df.groupby(group_col).agg(methods)


# TODO
class UdfAggFunc:
    """
    自定义聚合函数
    """
    # _class_config_param = None
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def mean(x: pd.Series):
        return np.mean(x)
    
    @staticmethod
    def std(x: pd.Series):
        return np.std(x)

    @staticmethod
    def median(x: pd.Series):
        return np.median(x)
    
    @staticmethod
    def variation_coefficient(x: pd.Series):
        mean = np.mean(x)
        if mean != 0:
            return np.std(x) / mean
        else:
            return np.nan
    
    @staticmethod
    def variance(x: pd.Series):
        return np.var(x)
    
    @staticmethod
    def skewness(x: pd.Series):
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return pd.Series.skew(x)


def mean(x: pd.Series) -> np.float:
    """
    均值

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    return np.mean(x)


def std(x: pd.Series) -> np.float:
    """
    标注差

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    return np.std(x)


def median(x: pd.Series) -> np.float:
    """
    中位数

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    return np.median(x)


def variation_coefficient(x: pd.Series) -> np.float:
    """
    变异系数

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def variance(x: pd.Series) -> np.float:
    """
    方差

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    return np.var(x)


def skewness(x: pd.Series) -> np.float:
    """
    偏度

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def kurtosis(x: pd.Seris) -> np.float:
    """
    峰度

    :param x: _description_
    :type x: pd.Seris
    :return: _description_
    :rtype: np.float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def large_std(x: pd.Series) -> np.float:
    """
    # TODO Large 标准差

    :param x: _description_
    :type x: pd.Series
    :return: _description_
    :rtype: np.float
    """
    if (np.max(x) - np.min(x)) != 0:
        return np.std(x) / (np.max(x) - np.min(x))
    else:
        return np.nan


def variance_std_ratio(x):
    """
    # TODO 方差标准率

    :param x: _description_
    :type x: _type_
    :return: _description_
    :rtype: _type_
    """
    y = np.var(x)
    if y != 0:
        return y / np.sqrt(y)
    else:
        return np.nan


def ratio_beyond_r_sigma(x, r):
    """
    _summary_

    :param x: _description_
    :type x: _type_
    :param r: _description_
    :type r: _type_
    :return: _description_
    :rtype: _type_
    """
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size


def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference


def has_duplicate_max(x):
    return np.sum(x == np.max(x)) >= 2


def has_duplicate_min(x):
    return np.sum(x == np.min(x)) >= 2


def has_duplicate(x):
    return x.size != np.unique(x).size


def count_duplicate_max(x):
    return np.sum(x == np.max(x))


def count_duplicate_min(x):
    return np.sum(x == np.min(x))


def count_duplicate(x):
    return x.size - np.unique(x).size


def sum_values(x):
    if len(x) == 0:
        return 0
    return np.sum(x)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def realized_abs_skew(series):
    return np.power(np.abs(np.sum(series**3)),1/3)


def realized_skew(series):
    return np.sign(np.sum(series**3))*np.power(np.abs(np.sum(series**3)),1/3)


def realized_vol_skew(series):
    return np.power(np.abs(np.sum(series**6)),1/6)


def realized_quarticity(series):
    return np.power(np.sum(series**4),1/4)


def count_unique(series):
    return len(np.unique(series))


def count(series):
    return series.size


def maximum_drawdown(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i])<1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j-k

def maximum_drawup(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    series = - series
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i])<1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j-k


def drawdown_duration(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j=k
    else:
        j = np.argmax(series[:i])
    return k-j


def drawup_duration(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    series=-series
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j=k
    else:
        j = np.argmax(series[:i])
    return k-j


def max_over_min(series):
    if len(series)<2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.max(series)/np.min(series)


def mean_n_absolute_max(x, number_of_maxima = 1):
    """ Calculates the arithmetic mean of the n absolute maximum values of the time series."""
    assert (
        number_of_maxima > 0
    ), f" number_of_maxima={number_of_maxima} which is not greater than 1"

    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.NaN


def count_above(x, t):
    if len(x)==0:
        return np.nan
    else:
        return np.sum(x >= t) / len(x)


def count_below(x, t):
    if len(x)==0:
        return np.nan
    else:
        return np.sum(x <= t) / len(x)


def number_peaks(x, n):
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(x, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(x, -i)[n:-n]
    return np.sum(res)


def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))


def mean_change(x):
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN


def mean_second_derivative_central(x):
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN


def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.NaN


def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))


def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0


def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0


def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size


def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size


def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


def percentage_of_reoccurring_values_to_all_values(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])


def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size


def sum_of_reoccurring_values(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def quantile(x, q):
    if len(x) == 0:
        return np.NaN
    return np.quantile(x, q)


def number_crossing_m(x, m):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > m
    return np.where(np.diff(positive))[0].size


def absolute_maximum(x):
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN


def value_count(x, value):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if np.isnan(value):
        return np.isnan(x).sum()
    else:
        return x[x == value].size


def range_count(x, min, max):
    return np.sum((x >= min) & (x < max))


def mean_diff(x):
    return np.nanmean(np.diff(x.values))




# 测试代码 main 函数
def main():
    df = pd.DataFrame({
        "group": [1, 1, 2, 2],
        "values": [4, 1, 1, 2],
        "values2": [0, 1, 1, 2],
    })
    # res = aggregation(df = df, methods = "sum", group_col = "group")
    # print(res)
    # res = df.groupby("group").agg([np.mean, np.std])
    # print(res)
    # udf_agg_func = UdfAggFunc()
    res = aggregation(
        df = df, 
        methods = [
            mean, 
            median,
        ], 
        group_col = "group"
    )
    print(res)


if __name__ == "__main__":
    main()

