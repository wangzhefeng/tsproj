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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import List

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


class day_of_week(TimeFeature):
    """Day of week encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek


# TODO
class weekday(TimeFeature):
    """Day of week encoded as value"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.weekday


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
            "name": ["year"],
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
            "name": ["year"],
            "func": [year],
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
            "name": ["day_of_month", "week_of_year"],
            "func": [day_of_month, week_of_year],
        },
        offsets.Day: {
            "name": ["day_of_week", "day_of_month", "day_of_year"],
            "func": [day_of_week, day_of_month, day_of_year],
        },
        offsets.BusinessDay: {
            "name": ["day_of_week", "day_of_month", "day_of_year"],
            "func": [day_of_week, day_of_month, day_of_year],
        },
        offsets.Hour: {
            "name": ["hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [hour_of_day, day_of_week, day_of_month, day_of_year],
        },
        offsets.Minute: {
            "name": ["minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        },
        offsets.Second: {
            "name": ["second_of_minute", "minute_of_hour", "hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "func": [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
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


def time_features_dl(dates, freq='h'):
    return np.vstack([
        feat(dates) 
        for feat in time_features_from_frequency_str_enc(freq)
    ])


def time_features_ml(df, time_col: str = "ds", freq: str = "h", timeenc = 0, result_type="all"):
    """
    > `time_features` takes in a `df` dataframe with a 'date' column 
    > and extracts the date down to `freq` where freq can be any of the 
    > following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
    > * s - [month, day, weekday, hour, *minute, *second]
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

    result_type: "all": all origin features of df and build features, 
         "ts": only timestamp features
         "ts_value": only timestamp features values
    """
    # 频率字符处理
    freq = freq.lower() 
    if timeenc == 0: 
        # 原始数据特征
        origin_feats = list(df.columns)
        # 时间特征编码
        df["year"] = df[time_col].apply(lambda row: row.year, 1)
        df['month'] = df[time_col].apply(lambda row: row.month, 1)
        df['day'] = df[time_col].apply(lambda row: row.day, 1)
        df['weekday'] = df[time_col].apply(lambda row: row.weekday(), 1)
        df['hour'] = df[time_col].apply(lambda row: row.hour, 1)
        df['minute'] = df[time_col].apply(lambda row: row.minute, 1)
        if freq == "5minute" or freq == "5min":
            df['minute'] = df.minute.map(lambda x: x // 5)
        if freq == "15minute" or freq == "15min":
            df['minute'] = df.minute.map(lambda x: x // 15)
        df['second'] = df[time_col].apply(lambda row: row.second, 1)
        if freq == "5seconds" or freq == "5sec" or freq == "5s":
            df['second'] = df.second.map(lambda x: x // 5)
        if freq == "15seconds" or freq == "15sec" or freq == "15s":
            df['second'] = df.second.map(lambda x: x // 15)
        # 频率字符处理
        freq_name_map = {
            "y": ["year", "1y", "y"],
            "m": ["month", "1m", "m"],
            "w": ["weekday", "1w", "w"],
            "d": ["day", "1d", "d"],
            "b": ["business day", "businessday", "1b", "b"],
            "h": ["hour", "1h", "h"],
            "t": ["minute", "1minute", "5minute", "15minute", "min", "1min", "5min", "15min"],
            "s": ["second", "1second", "5seconds", "10seconds", "15seconds", 
                  "sec", "1sec", "5sec", "10sec", "15sec", 
                  "s", "1s", "5s", "10s", "15s"]
        }
        for freq_name, freq_name_list in freq_name_map.items():
            if freq in freq_name_list:
                freq = freq_name
        # 特征筛选
        freq_feat_map = {
            'y': ['year'],
            'm': ['month'],
            'w': ['month', "day"],
            'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'],
            'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
            's': ['month', 'day', 'weekday', 'hour', 'minute', "second"],
        }
        freq_feats = freq_feat_map[freq]
        if result_type == "all":
            return df[origin_feats + freq_feats]
        elif result_type == "ts":
            return df[freq_feats]
        elif result_type == "ts_value":
            return df[freq_feats].values
    elif timeenc == 2:
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
        elif result_type == "ts_value":
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
        elif result_type == "ts_value":
            return np.vstack([
                feat_func(pd.to_datetime(df[time_col].values)) 
                for feat_func in feat_names_funcs.values()
            ]).transpose(1, 0)


def time_static_features(df, dt_is_index: bool = False, dt_name: str = None, dt_format: str = "%Y-%m-%d %H:%M:%S", features: List = []) -> pd.DataFrame:
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
    df = df.copy()
    
    # 日期时间特征处理
    dt_col = df.index if dt_is_index else df[dt_name]
    df["ds"] = pd.to_datetime(dt_col, format = dt_format)
    
    # 时间日期特征
    df["date"] = df["ds"].dt.date  # 日期
    df["time"] = df["ds"].dt.time  # 时间
    df["year"] = df["ds"].dt.year  # 年
    df["is_year_start"] = df["ds"].dt.is_year_start  # 是否年初
    df["is_year_end"] = df["ds"].dt.is_year_end  # 是否年末
    df["is_leap_year"] = df["ds"].dt.is_leap_year  # 是否是闰年
    df["quarter"] = df["ds"].dt.quarter  # 季度
    df["is_quarter_start"] = df["ds"].dt.is_quarter_start  # 是否季度初
    df["is_quarter_end"] = df["ds"].dt.is_quarter_end  # 是否季度末
    # TODO 季节
    # TODO 业务季度
    df["month"] = df["ds"].dt.month  # 月
    df["is_month_start"] = df["ds"].dt.is_month_start  # 是否月初
    df["is_month_end"] = df["ds"].dt.is_month_end  # 是否月末
    df["daysinmonth"] = df["ds"].dt.daysinmonth  # 每个月的天数
    # TODO 每个月中的工作日天数
    # TODO 每个月中的休假天数
    # TODO 是否夏时制
    df["weekofyear"] = df["ds"].dt.isocalendar().week  # 一年的第几周
    # TODO df["weekofmonth"] = df["ds"].apply(lambda x: x.weekofmonth(), 1)  # 一月中的第几周
    df["dayofyear"] = df["ds"].dt.dayofyear  # 一年的第几天
    df["dayofmonth"] = df["ds"].dt.day  # 日(一月中的第几天)
    df["dayofweek"] = df["ds"].dt.dayofweek  # 一周的第几天
    df["weekday"] = df["ds"].apply(lambda x: x.weekday(), 1)  # 周几
    df["is_weekend"] = df['dayofweek'].apply(is_weekend, 1)  # 是否周末
    # TODO df["is_holiday"] = df["ds"].apply(is_holiday, 1)  # 是否放假/是否工作日/是否节假日 
    # TODO 节假日连续天数
    # TODO 节假日前第 n 天
    # TODO 节假日第 n 天
    # TODO 节假日后第 n 天
    # TODO df["is_tiaoxiu"] = df["ds"].apply(is_tiaoxiu, 1)  # 是否调休
    df["hour"] = df["ds"].dt.hour  # 时(一天过去了几小时)
    df["minute"] = df["ds"].dt.minute  # 分
    # TODO df["past_minutes"] = df["ds"].apply(past_minutes, 1)  # 一天过去了几分钟
    df["second"] = df["ds"].dt.second  # 秒
    df["microsecond"] = df["ds"].dt.microsecond  # 微妙
    df["nanosecond"] = df["ds"].dt.nanosecond  # 纳秒
    # TODO df["time_period"] = df["ds"].apply(time_period, 1)  # 一天的哪个时间段
    # TODO df["day_high"] = df["hour"].apply(lambda x: 0 if 0 < x < 8 else 1, 1)  # 是否为高峰期
    # TODO df["is_work"] = df["hour"].apply(is_work, 1)  # 该时间点是否营业/上班
    
    # 删除时间列
    del df["ds"]
    
    # 数据特征筛选
    if features == []:
        selected_df = df
    else:
        selected_df = df[features]
     
    return selected_df


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
    data[col + '_sin'] = round(np.sin(2 * np.pi / n * data[col]), 6)
    data[col + '_cos'] = round(np.cos(2 * np.pi / n * data[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = ohe.fit_transform(data[col].values.reshape(-1, 1)).toarray()
        df = pd.DataFrame(X, columns = [col + '_' + str(int(i)) for i in range(X.shape[1])])
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
    res = time_features_ml(df=df, time_col="ds", freq="15min", timeenc=2, result_type="all")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-14 00:00:00", end="2024-11-15 00:46:00", freq="15min"),
        "unique_id": range(100),
        "y": range(100),
    })
    res = time_features_ml(df=df, time_col="ds", freq="15min", timeenc=0, result_type="all")
    with pd.option_context("display.max_columns", None):
        print(res)
        print("-" * 80)
    # ------------------------------
    # 
    # ------------------------------
    # res1 = time_static_features(df, dt_is_index=False, dt_name= "date", dt_format="%Y-%m-%d %H:%M:%S", features=[])
    # print(res1)
    # print(res1.columns)
    # print("-" * 80)

if __name__ == "__main__":
    main()
