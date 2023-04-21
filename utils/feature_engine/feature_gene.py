# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071720
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


__all__ = [
    "gen_time_features",
    "get_time_fe",
]


# python libraries
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from utils_func import is_weekend


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def gen_time_features(ts: pd.DataFrame, dt_format: str, dt_is_index: bool = False, dt_name: str = None, features: List = None) -> pd.DataFrame:
    """
    时间特征提取

    Args:
        data ([type]): 时间序列
        datetime_format ([type]): 时间特征日期时间格式
        datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
        datetime_name ([type], optional): 时间特征名称. Defaults to None.
        target_name ([type], optional): 目标特征名称. Defaults to False.
        features: 最后返回的特征名称列表
    """
    # 数据拷贝
    data = ts.copy()
    # 日期时间特征处理
    if dt_is_index:
        # df = df.set_index("季度")
        # df.index = pd.to_datetime(df.index)
        data["DT"] = data.index
        data["DT"] = pd.to_datetime(data["DT"], format = dt_format)
    else:
        data[dt_name] = pd.to_datetime(data[dt_name], format = dt_format)
        data["DT"] = data[dt_name]
    # 时间特征提取
    data["year"] = data["DT"].apply(lambda x: x.year)  # 年
    # data["year"] = data["DT"].dt.year
    data["quarter"] = data["DT"].apply(lambda x: x.quarter)  # 季度
    # data["quarter"] = data["DT"].dt.quarter
    data["month"] = data["DT"].apply(lambda x: x.month)  # 月
    # data["month"] = data["DT"].dt.month
    data["day"] = data["DT"].apply(lambda x: x.day)  # 日
    # data["day"] = data["DT"].dt.day
    data["hour"] = data["DT"].apply(lambda x: x.hour)  # 时
    # data["hour"] = data["DT"].dt.hour
    data["minute"] = data["DT"].apply(lambda x: x.minute)  # 分
    # data['minute'] = data['DT'].dt.minute
    data["second"] = data["DT"].apply(lambda x: x.second)  # 秒
    # data['second'] = data['DT'].dt.second
    data["dayofweek"] = data["DT"].apply(lambda x: x.dayofweek)  # 一周的第几天
    # data["dayofweek"] = data["DT"].dt.day_of_week
    data["dayofyear"] = data["DT"].apply(lambda x: x.dayofyear)  # 一年的第几天
    # data["dayofyear"] = data["DT"].dt.day_of_year
    data["weekofyear"] = data["DT"].apply(lambda x: x.weekofyear)  # 一年的第几周
    # data["weekofyear"] = data["DT"].dt.week
    data["is_year_start"] = data["DT"].apply(lambda x: x.is_year_start)  # 是否年初
    # data["is_year_start"] = data["DT"].dt.is_year_start
    data["is_year_end"] = data["DT"].apply(lambda x: x.is_year_end)  # 是否年末
    # data["is_year_end"] = data["DT"].dt.is_year_end
    data["is_quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)  # 是否季度初
    # data["is_quarter_start"] = data["DT"].dt.is_quarter_start
    data["is_quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)  # 是否季度末
    # data["is_quarter_start"] = data["DT"].dt.is_quarter_end
    data["is_month_start"] = data["DT"].apply(lambda x: x.is_month_start)  # 是否月初
    # data["is_month_start"] = data["DT"].dt.is_month_start
    data["is_month_end"] = data["DT"].apply(lambda x: x.is_month_end)  # 是否月末
    # data["is_month_end"] = data["DT"].dt.is_month_end
    data["weekend"] = data['dow'].apply(is_weekend)  # 是否周末
    data["day_high"] = data["hour"].apply(lambda x: 0 if 0 < x < 8 else 1)  # 是否为高峰期
    # 删除临时日期时间特征
    del data["DT"]
    # 特征选择
    if features is None:
        result = data
    else:
        result = data[features]
    
    return result 


def get_time_fe(data: pd.DataFrame, col: str, n: int, one_hot: bool = False, drop: bool = True):
    """
    构造时间特征

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
        df = pd.DataFrame(
            X, 
            columns = [col + '_' + str(int(i)) for i in range(X.shape[1])]
        )
        data = pd.concat([data, df], axis = 1)
        if drop:
            data = data.drop(col, axis = 1)

    return data


def gen_statistic_features(ts: pd.DataFrame) -> pd.DataFrame:
    pass


class Timeseries2Dataframe(object):

    def __init__(self):
        pass
    
    def timeseries2dataframe(self, data, n_lag = 1, n_fut = 1, selLag = None, selFut = None, dropnan = True):
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

    def gen_time_features(self, data, datetime_format, datetime_is_index = False, datetime_name = None, features = []):
        """
        时间特征提取

        Args:
            data ([type]): 时间序列
            datetime_format ([type]): 时间特征日期时间格式
            datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
            datetime_name ([type], optional): 时间特征名称. Defaults to None.
            features: 最后返回的特征名称列表
        """
        if datetime_is_index:
            data["DT"] = data.index
            data["DT"] = pd.to_datetime(data["DT"], format = datetime_format)
        else:
            data[datetime_name] = pd.to_datetime(data[datetime_name], format = datetime_format)
            data["DT"] = data[datetime_name]
        data["year"] = data["DT"].apply(lambda x: x.year)
        data["quarter"] = data["DT"].apply(lambda x: x.quarter)
        data["month"] = data["DT"].apply(lambda x: x.month)
        data["day"] = data["DT"].apply(lambda x: x.day)
        data["hour"] = data["DT"].apply(lambda x: x.hour)
        data["minute"] = None
        data["second"] = None
        data["dow"] = data["DT"].apply(lambda x: x.dayofweek)
        data["doy"] = data["DT"].apply(lambda x: x.dayofyear)
        data["woy"] = data["DT"].apply(lambda x: x.weekofyear)
        data["year_start"] = data["DT"].apply(lambda x: x.is_year_start)
        data["year_end"] = data["DT"].apply(lambda x: x.is_year_end)
        data["quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)
        data["quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)
        data["month_start"] = data["DT"].apply(lambda x: x.is_month_start)
        data["month_end"] = data["DT"].apply(lambda x: x.is_month_end)
        def applyer(row):
            """
            判断是否是周末
            """
            if row == 5 or row == 6:
                return 1
            else:
                return 0
        data["weekend"] = data['dow'].apply(applyer)
        del data["DT"]
        if features == []:
            selected_features = data
        else:
            selected_features = data[features]
        return selected_features

    def gen_time_features2(self, data, datetime_format, datetime_is_index = False, datetime_name = None, is_test = False):
        """
        时间特征提取

        Args:
            data ([type]): 时间序列
            datetime_format ([type]): 时间特征日期时间格式
            datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
            datetime_name ([type], optional): 时间特征特证名称. Defaults to None.
            is_test (bool, optional): 时间序列是否为测试数据. Defaults to False.
        """
        def applyer(row):
            """
            判断是否是周末
            """
            if row == 5 or row == 6:
                return 1
            else:
                return 0
        data[datetime_name] = pd.to_datetime(data[datetime_name], format = datetime_format)
        if datetime_is_index:
            data["DT"] = data.index
        else:
            data["DT"] = data[datetime_name]
        data["year"] = data["DT"].apply(lambda x: x.year)
        data['month'] = data["DT"].apply(lambda x: x.month)
        data['day'] = data["DT"].apply(lambda x: x.day)
        data["hour"] = data["DT"].apply(lambda x: x.hour)
        if is_test:
            pass
        else:
            data['dow'] = data["DT"].apply(lambda x: x.dayofweek)
            data['weekend'] = data['dow'].apply(applyer)
        del data["DT"]
        return data

    def gen_lag_features(self, data, cycle):
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

    def analysis_features_select(self, data, target_name):
        """
        random forest特征重要性分析
        """
        # data
        feature_names = data.columns.drop([target_name])
        feature_list = []
        for col in feature_names:
            feature_list.append(np.array(data[col]))
        X = np.array(feature_list).T
        y = np.array(np.array(data[target_name])).reshape(-1, 1)
        # rf
        rf_model = RandomForestRegressor(n_estimators = 500, random_state = 1)
        rf_model.fit(X, y)
        # show importance score
        print(rf_model.feature_importances_)
        # plot importance score
        ticks = [i for i in range(len(feature_names))]
        plt.bar(ticks, rf_model.feature_importances_)
        plt.xticks(ticks, feature_names)
        plt.show()

    def features_select(self, data, target_name):
        # data
        feature_names = data.columns.drop([target_name])
        feature_list = []
        for col in feature_names:
            feature_list.append(np.array(data[col]))
        X = np.array(feature_list).T
        y = np.array(np.array(data[target_name])).reshape(-1, 1)
        # rf
        rfe = RFE(RandomForestRegressor(n_estimators = 500, random_state = 1), n_features_to_select = 4)
        fit = rfe.fit(X, y)
        # report selected features
        print('Selected Features:')
        for i in range(len(fit.support_)):
            if fit.support_[i]:
                print(feature_names[i])
        # plot feature rank
        ticks = [i for i in range(len(feature_names))]
        plt.bar(ticks, fit.ranking_)
        plt.xticks(ticks, feature_names)
        plt.show()





# 测试代码 main 函数
def main():
    data = None
    data_df = gen_time_features(data)
    data_df = get_time_fe(data_df, 'hour', n = 24, one_hot = False, drop = False)
    data_df = get_time_fe(data_df, 'day', n = 31, one_hot = False, drop = True)
    data_df = get_time_fe(data_df, 'dayofweek', n = 7, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'season', n = 4, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'month', n = 12, one_hot = True, drop = True)
    data_df = get_time_fe(data_df, 'weekofyear', n = 53, one_hot = False, drop = True)

    # data
    series = pd.read_csv(
        "/Users/zfwang/machinelearning/datasets/car-sales.csv", 
        header = 0, 
        index_col = 0
    )
    # gen features
    ts2df = Timeseries2Dataframe()
    series = ts2df.timeseries2dataframe(
        data = series, 
        n_lag = 12, 
        n_fut = 0, 
        selLag = None, 
        selFut = None, 
        dropnan = True
    )
    ts2df.analysis_features_select(series, "Sales")
    ts2df.features_select(series, "Sales")

if __name__ == "__main__":
    main()
