# -*- coding: utf-8 -*-

# ***************************************************
# * File        : timefeatures.py
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

from models.ts_feature.tools import is_weekend

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def feature_engineer(df):
    """
    特征工程
    1. 时间戳特征
    2. 差分序列
    3. 同时刻风场、邻近风机的特均值、标准差
    """
    df["month"] = df.DATATIME.apply(lambda x: x.month, 1)
    df["day"] = df.DATATIME.apply(lambda x: x.day, 1)
    df["weekday"] = df.DATATIME.apply(lambda x: x.weekday(), 1)
    df["hour"] = df.DATATIME.apply(lambda x: x.hour, 1)
    df["minute"] = df.DATATIME.apply(lambda x: x.minute, 1)

    return df


def time_static_features(series, 
                         datetime_format: str = '%Y-%m-%d %H:%M:%S', 
                         datetime_is_index: bool = False, 
                         datetime_name: str = None, 
                         target_name: str = None,
                         features: List = []) -> pd.DataFrame:
    """
    时间特征提取

    Args:
        data ([type]): 时间序列
        datetime_format ([type]): 时间特征日期时间格式
        datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
        datetime_name ([type], optional): 时间特征名称. Defaults to None.
        features: 最后返回的特征名称列表
    """
    data = series.copy()
    # 日期时间特征处理
    if datetime_is_index:
        data["DT"] = data.index
        data["DT"] = pd.to_datetime(data["DT"], format = datetime_format)
    else:
        data[datetime_name] = pd.to_datetime(data[datetime_name], format = datetime_format)
        data["DT"] = data[datetime_name]
    # 时间日期特征
    data["date"] = data["DT"].apply(lambda x: x.date)  # 日期
    data["time"] = data["DT"].apply(lambda x: x.time)  # 时间
    data["year"] = data["DT"].apply(lambda x: x.year)  # 年
    data["is_year_start"] = data["DT"].apply(lambda x: x.is_year_start)  # 是否年初
    data["is_year_end"] = data["DT"].apply(lambda x: x.is_year_end)  # 是否年末
    data["is_leap_year"] = data["DT"].apply(lambda x: x.is_leap_year)  # 是否是闰年
    data["quarter"] = data["DT"].apply(lambda x: x.quarter)  # 季度
    data["is_quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)  # 是否季度初
    data["is_quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)  # 是否季度末
    # TODO 季节
    # TODO 业务季度
    data["month"] = data["DT"].apply(lambda x: x.month)  # 月
    data["is_month_start"] = data["DT"].apply(lambda x: x.is_month_start)  # 是否月初
    data["is_month_end"] = data["DT"].apply(lambda x: x.is_month_end)  # 是否月末
    data["daysinmonth"] = data["DT"].apply(lambda x: x.daysinmonth)  # 每个月的天数
    # TODO 每个月中的工作日天数
    # TODO 每个月中的休假天数
    # TODO 是否夏时制
    data["weekofyear"] = data["DT"].apply(lambda x: x.isocalendar().week)  # 一年的第几周
    # TODO 一月中的第几周
    data["dayofyear"] = data["DT"].apply(lambda x: x.dayofyear)  # 一年的第几天
    data["dayofmonth"] = data["DT"].apply(lambda x: x.day)  # 日(一月中的第几天)
    data["dayofweek"] = data["DT"].apply(lambda x: x.dayofweek)  # 一周的第几天
    data["is_weekend"] = data['dayofweek'].apply(is_weekend)  # 是否周末
    # TODO data["is_holiday"] = data["DT"].apply(is_holiday)  # 是否放假/是否工作日/是否节假日
    # TODO 节假日连续天数
    # TODO 节假日前第 n 天
    # TODO 节假日第 n 天
    # TODO 节假日后第 n 天
    # TODOdata["is_tiaoxiu"] = data["DT"].apply(is_tiaoxiu)  # 是否调休
    data["hour"] = data["DT"].apply(lambda x: x.hour)  # 时(一天过去了几分钟)
    data["minute"] = data["DT"].apply(lambda x: x.minute)  # 分
    # TODO data["past_minutes"] = data["DT"].apply(past_minutes)  # 一天过去了几分钟
    data["second"] = data["DT"].apply(lambda x: x.second)  # 秒
    data["microsecond"] = data["DT"].apply(lambda x: x.microsecond)  # 微妙
    data["nanosecond"] = data["DT"].apply(lambda x: x.nanosecond)  # 纳秒

    # TODO data["time_period"] = data["DT"].apply(time_period)  # 一天的哪个时间段
    data["day_high"] = data["hour"].apply(lambda x: 0 if 0 < x < 8 else 1)  # 是否为高峰期
    # TODO data["is_work"] = data["hour"].apply(is_work)  # 该时间点是否营业/上班
    
    del data["DT"]
    if features == []:
        selected_features = data
    else:
        selected_features = data[features]
        
    return selected_features


def time_dynamic_features(series, 
                          n_lag: int = 1, 
                          n_fut: int = 1, 
                          selLag = None, 
                          selFut = None, 
                          dropnan = True):
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
    data = series.copy()

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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


# TODO
def gen_lag_features(data, cycle):
    """
    时间序列滞后性特征
        - 二阶差分
    Args:
        data ([type]): 时间序列
        cycle ([type]): 时间序列周期
    """
    # 序列平稳化, 季节性差分
    series_diff = data.diff(cycle)
    series_diff = series_diff[cycle:]
    # 监督学习的特征
    for i in range(cycle, 0, -1):
        series_diff["t-" + str(i)] = series_diff.shift(i).values[:, 0]
    series_diff["t"] = series_diff.values[:, 0]
    series_diff = series_diff[cycle + 1:]
    return series_diff


class TimeFeature:

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    Second of Minute encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    Day of month encoded as value between [-0.5, 0.5]
    """
    # TODO
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    Day of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    Month of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    Week of year encoded as value between [-0.5, 0.5]
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Raises:
        RuntimeError: _description_

    Returns:
        List[TimeFeature]: _description_
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f'''
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
    '''
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc = 1, freq = "h"):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column 
    > and extracts the date down to `freq` where freq can be any of the 
    > following if `timeenc` is 0: 
    > * m - [month]
    > * w - [month]
    > * d - [month, day, weekday]
    > * b - [month, day, weekday]
    > * h - [month, day, weekday, hour]
    > * t - [month, day, weekday, hour, *minute]
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

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc == 0:
        dates['month'] = dates.date.apply(lambda row: row.month, 1)
        dates['day'] = dates.date.apply(lambda row: row.day, 1)
        dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
        dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
        dates['minute'] = dates.date.apply(lambda row: row.minute, 1)
        dates['minute'] = dates.minute.map(lambda x: x // 15)
        freq_map = {
            'y':[],
            'm':['month'],
            'w':['month'],
            'd':['month', 'day', 'weekday'],
            'b':['month', 'day', 'weekday'],
            'h':['month', 'day', 'weekday', 'hour'],
            't':['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc == 1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack([
            feat(dates) for feat in time_features_from_frequency_str(freq)
        ]).transpose(1, 0)




# 测试代码 main 函数
def main():
    # dates = pd.to_datetime([
    #     "2023-01-01 01:01:05", "2023-01-01 01:01:10", 
    #     "2023-01-01 01:01:15", "2023-01-01 01:01:20", 
    #     "2023-01-01 01:01:25"
    # ])
    # res = time_features(dates, freq = "5s")
    # print(res)
    # res2 = time_features_from_frequency_str("5s")
    # print(res2)

    # data = None
    # data_df = gen_time_features(data)
    # data_df = get_time_fe(data_df, 'hour', n = 24, one_hot = False, drop = False)
    # data_df = get_time_fe(data_df, 'day', n = 31, one_hot = False, drop = True)
    # data_df = get_time_fe(data_df, 'dayofweek', n = 7, one_hot = True, drop = True)
    # data_df = get_time_fe(data_df, 'season', n = 4, one_hot = True, drop = True)
    # data_df = get_time_fe(data_df, 'month', n = 12, one_hot = True, drop = True)
    # data_df = get_time_fe(data_df, 'weekofyear', n = 53, one_hot = False, drop = True)

    # # data
    # series = pd.read_csv(
    #     "/Users/zfwang/machinelearning/datasets/car-sales.csv", 
    #     header = 0, 
    #     index_col = 0
    # )
    # # gen features
    # ts2df = Timeseries2Dataframe()
    # series = ts2df.timeseries2dataframe(
    #     data = series, 
    #     n_lag = 12, 
    #     n_fut = 0, 
    #     selLag = None, 
    #     selFut = None, 
    #     dropnan = True
    # )
    # ts2df.analysis_features_select(series, "Sales")
    # ts2df.features_select(series, "Sales")
    pass

if __name__ == "__main__":
    main()
