# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TimeFeatures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041901
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
from typing import List
import copy

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import OneHotEncoder

from tools import is_weekend

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TimeFeature:

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


# TODO
class second_of_minute(TimeFeature):
    """Second of minute encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


# TODO
class minute_of_hour(TimeFeature):  
    """Minute of hour encoded as value"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class hour_of_day(TimeFeature):
    """Hour of day encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


# TODO
class day_of_week(TimeFeature):
    """Day of week encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek


# TODO
class weekday(TimeFeature):
    """Day of week encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.weekday()


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class day_of_month(TimeFeature):
    """Day of year encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.day


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class day_of_year(TimeFeature):
    """Day of year encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofyear


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class month_of_year(TimeFeature):
    """Month of year encoded as value"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.month


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


class week_of_year(TimeFeature):
    """Week of year encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.isocalendar().week

# TODO
class year(TimeFeature):
    """Year encoded as value as actual year"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.year


def time_features_from_frequency_str_enc(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Raises:
        RuntimeError: _description_

    Returns:
        List[TimeFeature]: _description_
    """
    # freq_str = freq_str.upper()
    features_by_offsets = {
        offsets.YearEnd: {
            "name": [],
            "func": [],
        },
        offsets.QuarterEnd: {
            "name": ["month_of_year"],
            "func": [MonthOfYear],
        },
        offsets.MonthEnd: {
            "name": ["month_of_year"],
            "func": [MonthOfYear],
        },
        offsets.Week: {
            "name": ["day_of_month", "week_of_year"],
            "func": [DayOfMonth, WeekOfYear],
        },
        offsets.Day: {
            "name": ["day_of_week", "day_of_month", "day_of_year"],
            "func": [DayOfWeek, DayOfMonth, DayOfYear],
        },
        offsets.BusinessDay: {
            "name": ["day_of_week", "day_of_month", "day_of_year"],
            "func": [DayOfWeek, DayOfMonth, DayOfYear],
        },
        offsets.Hour: {
            "name": ["hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        },
        offsets.Minute: {
            "name": ["minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        },
        offsets.Second: {
            "name": ["second_of_minute", "minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        },
    }
    # offset = pd.tseries.frequencies.to_offset(freq_str)
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            output = {
                name: cls()
                for name, cls in zip(feature_classes["name"], feature_classes["func"])
            }
            return output
    # raise RuntimeError(f"Unsupported frequency {freq_str}")
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features_from_frequency_str_notenc(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Raises:
        RuntimeError: _description_

    Returns:
        List[TimeFeature]: _description_
    """
    # freq_str = freq_str.upper()
    features_by_offsets = {
        offsets.YearEnd: {
            "name": [],
            "func": [],
        },
        offsets.QuarterEnd: {
            "name": ["month_of_year"],
            "func": [month_of_year],
        },
        offsets.MonthEnd: {
            "name": ["month_of_year"],
            "func": [month_of_year],
        },
        offsets.Week: {
            "name": ["week_of_year", "month_of_year"],
            "func": [week_of_year, month_of_year],
        },
        offsets.Day: {
            "name": ["day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"],
            "func": [day_of_week, day_of_month, day_of_year, week_of_year, month_of_year],
        },
        offsets.BusinessDay: {
            "name": ["day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"],
            "func": [day_of_week, day_of_month, day_of_year, week_of_year, month_of_year],
        },
        offsets.Hour: {
            "name": ["hour_of_day", "day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"],
            "func": [hour_of_day, day_of_week, day_of_month, day_of_year, week_of_year, month_of_year],
        },
        offsets.Minute: {
            "name": ["minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"],
            "func": [minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year, week_of_year, month_of_year],
        },
        offsets.Second: {
            "name": ["second_of_minute", "minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"],
            "func": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year, week_of_year, month_of_year],
        },
    }
    # offset = pd.tseries.frequencies.to_offset(freq_str)
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            output = {
                name: cls()
                for name, cls in zip(feature_classes["name"], feature_classes["func"])
            }
            return output
    # raise RuntimeError(f"Unsupported frequency {freq_str}")
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(data, time_col: str = "ds", freq: str = "h", timeenc = 0, result_type="all"):
    """
    > `time_features` takes in a `df` dataframe with a 'date' column 
    > and extracts the date down to `freq` where freq can be any of the 
    > following if `timeenc` is 0: 
    > * q - [month(monthofyear)]
    > * m - [month(monthofyear)]
    > * w - [month(monthofyear), weekofyear]
    > * d - [month(monthofyear), weekofyear, dayofyear, day(dayofmonth), weekday(dayofweek)]
    > * b - [month(monthofyear), weekofyear, dayofyear, day(dayofmonth), weekday(dayofweek)]
    > * h - [month(monthofyear), weekofyear, dayofyear, day(dayofmonth), weekday(dayofweek), hour]
    > * t - [month(monthofyear), weekofyear, dayofyear, day(dayofmonth), weekday(dayofweek), hour, *minute]
    > * s - [month(monthofyear), weekofyear, dayofyear, day(dayofmonth), weekday(dayofweek), hour, *minute, *second]
    > 
    > If `timeenc` is 1, a similar, but different list of `freq` values 
    > are supported (all encoded between [-0.5 and 0.5]): 
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-11 corresponding to the 5 minute period it falls into.
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    *second returns a number from 0-11 corresponding to the 5 second period it falls into.
    *second returns a number from 0-3 corresponding to the 15 second period it falls into.

    result_type: 
        - "all": all origin features of df and build features, 
        - "ts": only timestamp features
        - "ts_values": only timestamp features values
    """
    # 数据备份
    df = copy.deepcopy(data)
    # 频率字符处理
    freq = freq.lower()
    if timeenc == 0:
        feat_names_funcs = time_features_from_frequency_str_notenc(freq)
        if result_type == "all":
            for feat_name, feat_func in feat_names_funcs.items():
                df[feat_name] = feat_func(pd.to_datetime(df[time_col].values))
            return df
        elif result_type == "ts":
            for feat_name, feat_func in feat_names_funcs.items():
                df[feat_name] = feat_func(pd.to_datetime(df[time_col].values))
            selected_feats = list(feat_names_funcs.keys())
            return df[selected_feats]
        elif result_type == "ts_values":
            return np.vstack([
                feat_func(pd.to_datetime(df[time_col].values)) 
                for feat_func in feat_names_funcs.values()
            ]).transpose(1, 0)
    elif timeenc == 1:
        feat_names_funcs = time_features_from_frequency_str_enc(freq)
        if result_type == "all":
            for feat_name, feat_func in feat_names_funcs.items():
                df[feat_name] = feat_func(pd.to_datetime(df[time_col].values))
            return df
        elif result_type == "ts":
            for feat_name, feat_func in feat_names_funcs.items():
                df[feat_name] = feat_func(pd.to_datetime(df[time_col].values))
            selected_feats = list(feat_names_funcs.keys())
            return df[selected_feats]
        elif result_type == "ts_values":
            return np.vstack([
                feat_func(pd.to_datetime(df[time_col].values)) 
                for feat_func in feat_names_funcs.values()
            ]).transpose(1, 0)


# TODO
def time_static_features(data, time_col: str, freq: str, timeenc: int = 0, result_type: str = "all"):
    """
    时间特征提取

    Args:
        df ([type]): 时间序列
        datetime_format ([type]): 时间特征日期时间格式
        datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
        datetime_name ([type], optional): 时间特征名称. Defaults to None.
        features: 最后返回的特征名称列表
    """
    # 数据备份
    df = copy.deepcopy(data)
    # 频率字符处理
    freq = freq.lower()
    # 原始数据特征
    origin_feats = list(df.columns)
    # 时间特征编码
    df["date"] = df[time_col].apply(lambda row: row.date(), 1)  # 日期
    df["time"] = df[time_col].apply(lambda row: row.time(), 1)  # 时间
    
    df["year"] = df[time_col].apply(lambda row: row.year, 1)  # 年
    df["is_year_start"] = df[time_col].apply(lambda row: row.is_year_start, 1)  # 是否年初
    df["is_year_end"] = df[time_col].apply(lambda row: row.is_year_end, 1)  # 是否年末
    df["is_leap_year"] = df[time_col].apply(lambda row: row.is_leap_year, 1)  # 是否是闰年
    
    df["quarter"] = df[time_col].apply(lambda row: row.quarter, 1)  # 季度
    df["is_quarter_start"] = df[time_col].apply(lambda row: row.is_quarter_start, 1)  # 是否季度初
    df["is_quarter_end"] = df[time_col].apply(lambda row: row.is_quarter_end, 1)  # 是否季度末
    
    df["month"] = df[time_col].apply(lambda row: row.month, 1)  # 月
    df["is_month_start"] = df[time_col].apply(lambda row: row.is_month_start, 1)  # 是否月初
    df["is_month_end"] = df[time_col].apply(lambda row: row.is_month_end, 1)  # 是否月末
    
    df["weekofyear"] = df[time_col].apply(lambda row: row.weekofyear, 1)  # 周
    # TODO df["weekofmonth"] = df["ds"].apply(lambda x: x.weekofmonth(), 1)  # 一月中的第几周
    # df["is_week_start"] = df[time_col].apply(lambda row: row.is_week_start, 1)  # 是否周初
    # df["is_week_end"] = df[time_col].apply(lambda row: row.is_week_end, 1)  # 是否周末

    # df["days_in_year"] = df[time_col].apply(lambda row: row.daysinyear, 1)  # 每年天数
    df["days_in_month"] = df[time_col].apply(lambda row: row.daysinmonth, 1)  # 每月天数
    df["day_of_year"] = df[time_col].apply(lambda row: row.dayofyear, 1)  # 一年中的第几天
    df["day_of_month"] = df[time_col].apply(lambda row: row.day, 1)  # 一月中的第几天
    df["day_of_week"] = df[time_col].apply(lambda row: row.dayofweek, 1)  # 一周中的第几天
    df["weekday"] = df[time_col].apply(lambda row: row.weekday(), 1)  # 一周中的第几天
    # df["is_weekend"] = df[time_col].apply(lambda row: row.is_weekend >= 5, 1)  # 是否是周末

    # TODO df["is_holiday"] = df["ds"].apply(is_holiday, 1)  # 是否放假/是否工作日/是否节假日 
    # TODO 节假日连续天数
    # TODO 节假日前第 n 天
    # TODO 节假日第 n 天
    # TODO 节假日后第 n 天
    # TODO df["is_tiaoxiu"] = df["ds"].apply(is_tiaoxiu, 1)  # 是否调休

    df["hour"] = df[time_col].apply(lambda row: row.hour, 1)
    
    df["minute"] = df[time_col].apply(lambda row: row.minute, 1)
    if freq == "5minute" or freq == "5min":
        df["minute"] = df.minute.map(lambda x: x // 5)
    if freq == "15minute" or freq == "15min":
        df["minute"] = df.minute.map(lambda x: x // 15)
    
    # TODO df["past_minutes"] = df["ds"].apply(past_minutes, 1)  # 一天过去了几分钟
    df["second"] = df[time_col].apply(lambda row: row.second, 1)
    if freq == "5seconds" or freq == "5sec" or freq == "5s":
        df["second"] = df.second.map(lambda x: x // 5)
    if freq == "15seconds" or freq == "15sec" or freq == "15s":
        df["second"] = df.second.map(lambda x: x // 15) 
    df["microsecond"] = df[time_col].dt.microsecond  # 微妙
    df["nanosecond"] = df[time_col].dt.nanosecond  # 纳秒
    # TODO df["time_period"] = df["ds"].apply(time_period, 1)  # 一天的哪个时间段
    # TODO df["day_high"] = df["hour"].apply(lambda x: 0 if 0 < x < 8 else 1, 1)  # 是否为高峰期
    # TODO df["is_work"] = df["hour"].apply(is_work, 1)  # 该时间点是否营业/上班
    # 频率/频率名称/特征筛选
    freq_name_feat_map = {
        "y": {
            "name": ["year", "1y", "y"],
            "feat": [],
        },
        "q": {
            "name": ["quarter", "1q", "q"],
            "feat": ["month"],
        },
        "m": {
            "name": ["month", "1m", "m"],
            "feat": ["month"],
        },
        "w": {
            "name": ["weekday", "1w", "w"], 
            "feat": ["month", "day"],
        },
        "d": {
            "name": ["day", "1d", "d"],
            "feat": ["month", "day", "weekday"],
        },
        "b": {
            "name": ["business day", "businessday", "1b", "b"],
            "feat": ["month", "day", "weekday"],
        },
        "h": {
            "name": ["hour", "1h", "h"],
            "feat": ["month", "day", "weekday", "hour"],
        },
        "t": {
            "name": ["minute", "1minute", "5minute", "15minute", "min", "1min", "5min", "15min"],
            "feat": ["month", "day", "weekday", "hour", "minute"],
        },
        "s": {
            "name": [
                "second", "1second", "5seconds", "10seconds", "15seconds", 
                "sec", "1sec", "5sec", "10sec", "15sec", 
                "s", "1s", "5s", "10s", "15s"
            ],
            "feat": ["month", "day", "weekday", "hour", "minute", "second"],
        },
    }
    for freq_str, name_feat in freq_name_feat_map.items():
        if freq in name_feat["name"]:
            freq_feats = name_feat["feat"]
    # 特征输出
    if result_type == "all":
        return df[origin_feats + freq_feats]
    elif result_type == "ts":
        return df[freq_feats]
    elif result_type == "ts_value":
        return df[freq_feats].values

# TODO
def time_dynamic_features(df, n_lag: int = 1, n_fut: int = 1, selLag = None, selFut = None, dropnan = True):
    """
    Converts a time series to a supervised learning data set by adding time-shifted 
    prior and future period data as input or output (i.e., target result) columns for each period.

    Params:
        data: a series of periodic attributes as a list or NumPy array.
        n_lag: number of PRIOR periods to lag as input (X); generates: Xa(t-1), Xa(t-2); min = 0 --> nothing lagged.
        n_fut: number of FUTURE periods to add as target output (y); generates Yout(t+1); min = 0 --> no future periods.
        selLag: only copy these specific PRIOR period attributes; default = None; EX: ['Xa', 'Xb' ].
        selFut: only copy these specific FUTURE period attributes; default = None; EX: ['rslt', 'xx'].
        dropnan: True = drop rows with NaN values; default = True.
    Returns:
        a Pandas DataFrame of time series data organized for supervised learning.
    NOTES:
        (1) The current period's data is always included in the output.
        (2) A suffix is added to the original column names to indicate a relative time reference: 
            e.g.(t) is the current period; 
                (t-2) is from two periods in the past; 
                (t+1) is from the next period.
        (3) This is an extension of Jason Brownlee's series_to_supervised() function, customized for MFI use
    """
    # 数据备份
    df = df.copy()
    # 特征个数
    n_vars = 1 if type(df) is list else df.shape[1]
    # 转换为 pandas.DataFrame
    df = pd.DataFrame(df)
    # 特征名称
    origNames = df.columns

    cols, names = list(), list()
    # include all current period attributes
    cols.append(df.shift(0))
    names += [("%s" % origNames[j]) for j in range(n_vars)]
    # ----------------------------------------------------
    # lag any past period attributes (t-n_lag, ..., t-1)
    # ----------------------------------------------------
    n_lag = max(0, n_lag)
    # input sequence (t-n, ..., t-1)
    for i in range(n_lag, 0, -1):
        suffix = "(t-%d)" % i
        if (selLag is None):
            cols.append(df.shift(i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
        else:
            for var in (selLag):
                cols.append(df[var].shift(i))
                names += [("%s%s" % (var, suffix))]
    # ----------------------------------------------------
    # include future period attributes (t+1, ..., t+n_fut)
    # ----------------------------------------------------
    n_fut = max(0, n_fut)
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_fut + 1):
        suffix = "(t+%d)" % i
        if (selFut is None):
            cols.append(df.shift(-i))
            names += [("%s%s" % (origNames[j], suffix)) for j in range(n_vars)]
        else:
            for var in (selFut):
                cols.append(df[var].shift(-i))
                names += [("%s%s" % (var, suffix))]
    # ----------------------------------------------------
    # put it all together
    # ----------------------------------------------------
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    # ----------------------------------------------------
    # drop rows with NaN values
    # ----------------------------------------------------
    if dropnan:
        agg.dropna(inplace = True)

    return agg


# TODO
def get_time_sin_cos(data: pd.DataFrame, col: str, n: int, one_hot: bool = False, drop: bool = True):
    """
    构造时间特征
    取 cos/sin 将数值的首位衔接起来, 比如说 23 点与 0 点很近, 星期一和星期天很近

    Args:
        data (_type_): _description_
        col (_type_): column name
        n (_type_): 时间周期
        one_hot (bool, optional): _description_. Defaults to False.
        drop (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    data[col + "_sin"] = round(np.sin(2 * np.pi / n * data[col]), 6)
    data[col + "_cos"] = round(np.cos(2 * np.pi / n * data[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = ohe.fit_transform(data[col].values.reshape(-1, 1)).toarray()
        df = pd.DataFrame(X, columns = [col + "_" + str(int(i)) for i in range(X.shape[1])])
        data = pd.concat([data, df], axis = 1)
        if drop:
            data = data.drop(col, axis = 1)

    return data




# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-14 00:00:00", end="2024-11-15 00:46:00", freq="15min"),
        "unique_id": range(100),
        "y": range(100),
    })
    # ------------------------------
    # 
    # ------------------------------
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=0, result_type="all")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=0, result_type="ts")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=0, result_type="ts_values")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    # ------------------------------
    # 
    # ------------------------------
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=1, result_type="all")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=1, result_type="ts")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    
    res = time_features(data=df, time_col="ds", freq="15min", timeenc=1, result_type="ts_values")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)

if __name__ == "__main__":
    main()
