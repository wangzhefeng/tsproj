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
import os
import gc
import random
import glob
from joblib import Parallel, delayed
import logging
import itertools
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DEBUG = False


def Aggregation(df: pd.DataFrame, methods: str = None, group_col: str = None):
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

    Args:
        df (pd.DataFrame): _description_
        methods (str, optional): _description_. Defaults to None.
        group_col (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if methods is None:
        logging.info(f"{LOGGING_LABEL} methods is None, please input correct 'methods' parameter.")
        return None
    if group_col is None:
        logging.info(f"{LOGGING_LABEL} group column is None, please input correct 'group_col' parameter.")
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
        # df.groupby(group_col).agg([np.mean, np.std])
        return df.groupby(group_col).agg(methods)
    elif isinstance(methods, dict):
        # df.groupby(group_col).agg({col1: ["mean", "median",...], col2: ["mean", "std",...]})      
        return df.groupby(group_col).agg(methods)


def length(x) -> int:
    """
    序列长度
    """
    return len(x)


def mean(x) -> np.float32:
    """
    均值
    """
    return np.mean(x)


def std(x) -> np.float32:
    """
    标准差
    """
    return np.std(x)


def large_std(x) -> np.float32:
    """
    Large 标准差
    """
    if (np.max(x) - np.min(x)) != 0:
        return np.std(x) / (np.max(x) - np.min(x))
    else:
        return np.nan


def quantile(x, q):
    if len(x) == 0:
        return np.NaN
    return np.quantile(x, q)


def median(x) -> np.float32:
    """
    中位数 
    """
    return np.median(x)


def variation_coefficient(x) -> np.float32:
    """
    变异系数
    """
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def variance_std_ratio(x) -> np.float32:
    """
    方差标准率
    """
    y = np.var(x)
    if y != 0:
        return y / np.sqrt(y)
    else:
        return np.nan


def variance(x) -> np.float32:
    """
    方差
    """
    return np.var(x)


def skewness(x) -> np.float32:
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def kurtosis(x: pd.Series) -> np.float32:
    """
    峰度
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def ratio_beyond_r_sigma(x, r):
    """
    # TODO
    """
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size


def range_ratio(x):
    """
    # TODO
    """
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference


def has_duplicate_min(x):
    """
    # TODO
    """
    return np.sum(x == np.min(x)) >= 2


def has_duplicate_max(x):
    """
    # TODO
    """
    return np.sum(x == np.max(x)) >= 2


def has_duplicate(x):
    """
    # TODO
    """
    return x.size != np.unique(x).size


def count_duplicate_min(x):
    """
    # TODO
    """
    return np.sum(x == np.min(x))


def count_duplicate_max(x):
    """
    # TODO
    """
    return np.sum(x == np.max(x))


def count_duplicate(x):
    """
    # TODO
    """
    return x.size - np.unique(x).size


def sum_values(x):
    """
    # TODO
    """
    if len(x) == 0:
        return 0
    return np.sum(x)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 


def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


def realized_abs_skew(series):
    return np.power(np.abs(np.sum(series ** 3)), 1 / 3)


def realized_skew(series):
    return np.sign(np.sum(series ** 3)) * np.power(np.abs(np.sum(series ** 3)), 1 / 3)


def realized_vol_skew(series):
    return np.power(np.abs(np.sum(series ** 6)), 1 / 6)


def realized_quarticity(series):
    return np.power(np.sum(series ** 4), 1 / 4)


def count_unique(series):
    return len(np.unique(series))


def count(series):
    return series.size


def maximum_drawdown(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def maximum_drawup(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    series = - series
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def drawdown_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j=k
    else:
        j = np.argmax(series[:i])
    return k - j


def drawup_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    series =- series
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j = k
    else:
        j = np.argmax(series[:i])
    return k - j


def max_over_min(series):
    if len(series) < 2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.max(series) / np.min(series)


def max_over_min_sq(series):
    if len(series) < 2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.square(np.max(series) / np.min(series))


def mean_n_absolute_max(x, number_of_maxima = 1):
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.
    """
    assert (number_of_maxima > 0), f" number_of_maxima={number_of_maxima} which is not greater than 1"
    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]
    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.NaN


def count_above(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x >= t) / len(x)


def count_below(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x <= t) / len(x)


def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. 
    A peak of support n is defined as a subsequence of x where a value occurs, 
    which is bigger than its n neighbours to the left and to the right.
    """
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

    
def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))


def mean_change(x):
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN


def mean_second_derivative_central(x):
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN


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


def number_crossing_m(x, m):
    """
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    positive = x > m
    return np.where(np.diff(positive))[0].size


def maximum(x):
    return np.max(x)


def absolute_maximum(x):
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN


def minimum(x):
    return np.min(x)


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


def _roll(a, shift):
    """
    Roll 1D array elements. Improves the performance of numpy.roll()
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version 
    of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len
    """
    return [
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def _into_subchunks(x, subchunk_length, every_n = 1):
    """
    Split the time series x into subwindows of length "subchunk_length", 
    starting every "every_n".
    """
    len_x = len(x)
    assert subchunk_length > 1
    assert every_n > 0
    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis = 0) + np.expand_dims(shift_starts, axis = 1)
    return np.asarray(x)[indexer]


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = (
                func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
            )
        return func

    return decorate_func


# ------------------------------
# Lambda functions to facilitate application
# ------------------------------
count_above_0 = lambda x: count_above(x, 0)
count_above_0.__name__ = 'count_above_0'

count_below_0 = lambda x: count_below(x, 0)
count_below_0.__name__ = 'count_below_0'

value_count_0 = lambda x: value_count(x, 0)
value_count_0.__name__ = 'value_count_0'

count_near_0 = lambda x: range_count(x, -0.00001, 0.00001)
count_near_0.__name__ = 'count_near_0_0'

ratio_beyond_01_sigma = lambda x: ratio_beyond_r_sigma(x, 0.1)
ratio_beyond_01_sigma.__name__ = 'ratio_beyond_01_sigma'

ratio_beyond_02_sigma = lambda x: ratio_beyond_r_sigma(x, 0.2)
ratio_beyond_02_sigma.__name__ = 'ratio_beyond_02_sigma'

ratio_beyond_03_sigma = lambda x: ratio_beyond_r_sigma(x, 0.3)
ratio_beyond_03_sigma.__name__ = 'ratio_beyond_03_sigma'

number_crossing_0 = lambda x: number_crossing_m(x, 0)
number_crossing_0.__name__ = 'number_crossing_0'

quantile_01 = lambda x: quantile(x, 0.1)
quantile_01.__name__ = 'quantile_01'

quantile_025 = lambda x: quantile(x, 0.25)
quantile_025.__name__ = 'quantile_025'

quantile_075 = lambda x: quantile(x, 0.75)
quantile_075.__name__ = 'quantile_075'

quantile_09 = lambda x: quantile(x, 0.9)
quantile_09.__name__ = 'quantile_09'

number_peaks_2 = lambda x: number_peaks(x,2)
number_peaks_2.__name__ = 'number_peaks_2'

mean_n_absolute_max_2 = lambda x: mean_n_absolute_max(x, 2)
mean_n_absolute_max_2.__name__ = 'mean_n_absolute_max_2'

number_peaks_5 = lambda x: number_peaks(x, 5)
number_peaks_5.__name__ = 'number_peaks_5'

mean_n_absolute_max_5 = lambda x: mean_n_absolute_max(x, 5)
mean_n_absolute_max_5.__name__ = 'mean_n_absolute_max_5'

number_peaks_10 = lambda x: number_peaks(x, 10)
number_peaks_10.__name__ = 'number_peaks_10'

mean_n_absolute_max_10 = lambda x: mean_n_absolute_max(x, 10)
mean_n_absolute_max_10.__name__ = 'mean_n_absolute_max_10'

get_first = lambda x: x.iloc[0]
get_first.__name__ = 'get_first'

get_last = lambda x: x.iloc[-1]
get_last.__name__ = 'get_last'


# ------------------------------
# 
# ------------------------------
base_stats = [
    length,
    sum,
    mean,
    std,
    variation_coefficient,
    variance,
    skewness,
    kurtosis,
]
higher_order_stats = [
    abs_energy,
    root_mean_square,
    sum_values,
    realized_volatility,
    realized_abs_skew,
    realized_skew,
    realized_vol_skew,
    realized_quarticity,
]
min_median_max = [
    minimum, 
    median, 
    maximum
]
additional_quantiles = [
    quantile_01,
    quantile_025,
    quantile_075,
    quantile_09,
]
other_min_max = [
    absolute_maximum, 
    max_over_min, 
    max_over_min_sq
]
min_max_positions = [
    last_location_of_maximum, 
    first_location_of_maximum, 
    last_location_of_minimum, 
    first_location_of_minimum
]
peaks = [
    number_peaks_2, 
    mean_n_absolute_max_2, 
    number_peaks_5, 
    mean_n_absolute_max_5, 
    number_peaks_10, 
    mean_n_absolute_max_10
]
counts = [
    count_unique, 
    count, 
    count_above_0, 
    count_below_0, 
    value_count_0, 
    count_near_0
]
variations = [
    mean_abs_change, 
    mean_change, 
    mean_second_derivative_central, 
    absolute_sum_of_changes, 
    number_crossing_0
]
ranges = [
    variance_std_ratio, 
    ratio_beyond_01_sigma, 
    ratio_beyond_02_sigma, 
    ratio_beyond_03_sigma, 
    large_std, range_ratio
]
get_first_last = [
    get_first, 
    get_last
]
reoccuring_values = [
    count_above_mean, 
    count_below_mean,
    percentage_of_reoccurring_values_to_all_values, 
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    sum_of_reoccurring_values,
    sum_of_reoccurring_data_points,
    ratio_value_number_to_time_series_length
]
count_duplicate = [
    count_duplicate, 
    count_duplicate_min, 
    count_duplicate_max
]

all_functions = (
    base_stats
    + higher_order_stats
    + min_median_max
    + additional_quantiles
    + other_min_max
    + min_max_positions
    + peaks
    + counts
    + variations
    + ranges
    + get_first_last
    # + reoccuring_values
    # + count_duplicate
)


def prepare_df(df, features_names: List):
    feature_dict = {
        func: all_functions
        for func in features_names
    }
    df_features = df.groupby("sequence").agg(feature_dict)
    df_features.columns = ["_".join(col) for col in df_features.columns]
    map_sequence_subject = df.groupby(["sequence"]).subject.min()
    map_subject_count = df.groupby(["sequence"]).subject.min().value_counts()
    df_features["count_sequence"] = df_features.index.map(
        map_sequence_subject.map(map_subject_count)
    )
    
    return df_features




# 测试代码 main 函数
def main():
    # ------------------------------
    # Aggregation
    # ------------------------------
    # df = pd.DataFrame({
    #     "group": [1, 1, 2, 2],
    #     "values": [4, 1, 1, 2],
    #     "values2": [0, 1, 1, 2],
    # })
    # res = Aggregation(df = df, methods = "sum", group_col = "group")
    # print(res)
    # ------------------------------
    # feature engineering
    # ------------------------------
    # data
    train = pd.read_csv("./data/train.csv")
    train_labels = pd.read_csv("./data/train_labels.csv")
    test = pd.read_csv("./data/test.csv")
    if DEBUG:
        train = train[train.sequence.isin(train.sequence.unique()[:100])]
        train_labels = train_labels.iloc[:100]
    print(train.head())
    # feature names
    features = ["sensor_" + str(i).zfill(2) for i in range(13)]
    print(features)

    # build aggregation dict
    feature_dict = {
        func: all_functions for func in features
    }
    
    # prepare feaures
    train_features = prepare_df(train, feature_dict)
    test_features = prepare_df(test, feature_dict)
    train_features.to_parquet("./data/train.parquet")
    test_features.to_parquet("./data/test.parquet")

    # model training
    clf = lgb.LGBMClassifier()
    clf.fit(train_features, train_labels.state)

    # feature importance
    feature_importance = pd.DataFrame(
        sorted(zip(clf.feature_importances_, train_features.columns)), 
        columns = ['Value','Feature']
    )
    plt.figure(figsize = (20, 10))
    sns.barplot(
        x = "Value", 
        y = "Feature", 
        data = feature_importance.sort_values(by = "Value", ascending = False).iloc[:50]
    )
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
