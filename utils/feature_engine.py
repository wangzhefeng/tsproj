# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feature_engine.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-09
# * Version     : 1.0.120914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import math
from typing import List

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO 根据数据频率构造时间特征
def extend_datetime_features(df: pd.DataFrame, feature_names: str=None):
    """
    增加时间特征
    """
    if feature_names == []:
        return df, feature_names
    else:
        # copy df_history
        df_history_with_datetime = df.copy()
        # 构造日期时间特征
        feature_map = {
            "minute": lambda x: x.minute,
            "hour": lambda x: x.hour,
            "day": lambda x: x.day,
            "weekday": lambda x: x.weekday(),
            "week": lambda x: x.week,
            "day_of_week": lambda x: x.dayofweek,
            "week_of_year": lambda x: x.weekofyear,
            "month": lambda x: x.month,
            "days_in_month": lambda x: x.daysinmonth,
            "quarter": lambda x: x.quarter,
            "day_of_year": lambda x: x.dayofyear,
            "year": lambda x: x.year,
        }
        for feature_name in feature_names:
            func = feature_map[feature_name]
            df_history_with_datetime[f"datetime_{feature_name}"] = df_history_with_datetime["ds"].apply(func)
        # 日期时间特征
        datetime_features = [
            col for col in df_history_with_datetime.columns 
            if col.startswith("datetime")
        ]
        return df_history_with_datetime, datetime_features


def extend_datetype_features(df: pd.DataFrame, df_date: pd.DataFrame=None):
    """
    增加日期类型特征：
    1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
    """ 
    if df_date is None:
        return df, []
    else:
        # history data copy
        df_history_with_datetype = df.copy()
        df_date_history_copy = df_date.copy()
        # 构造日期特征
        df_history_with_datetype["date"] = df_history_with_datetype["ds"].apply(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        # 匹配日期类型特征
        df_history_with_datetype["date_type"] = df_history_with_datetype["date"].map(
            df_date_history_copy.set_index("date")["date_type"]
        )
        # 日期类型特征
        date_features = ["date_type"]
        return df_history_with_datetype, date_features


def extend_weather_features(df_history: pd.DataFrame, df_weather_history: pd.DataFrame=None):
    """
    处理天气特征
    """
    if df_weather_history is None:
        return df_history, []
    else:
        # history data copy
        df_history_with_weather = df_history.copy()
        df_weather_copy = df_weather_history.copy()
        # 筛选天气预测数据
        weather_features_raw = ["rt_ssr", "rt_ws10", "rt_tt2", "rt_dt", "rt_ps", "rt_rain"]
        df_weather_copy = df_weather_copy[["ds"] + weather_features_raw]
        # 删除含空值的行
        df_weather_copy.dropna(inplace=True, ignore_index=True)
        # 将除了 ds 的列转为 float 类型
        for col in weather_features_raw:
            df_weather_copy[col] = df_weather_copy[col].apply(lambda x: float(x))
        # 计算相对湿度
        df_weather_copy["cal_rh"] = np.nan
        for i in df_weather_copy.index:
            if (df_weather_copy.loc[i, "rt_tt2"] is not np.nan
                and df_weather_copy.loc[i, "rt_dt"] is not np.nan):
                # 通过温度和露点温度计算相对湿度
                temp = (
                    math.exp(17.2693
                        * (df_weather_copy.loc[i, "rt_dt"] - 273.15)
                        / (df_weather_copy.loc[i, "rt_dt"] - 35.86))
                    / math.exp(17.2693
                        * (df_weather_copy.loc[i, "rt_tt2"] - 273.15)
                        / (df_weather_copy.loc[i, "rt_tt2"] - 35.86))
                    * 100
                )
                if temp < 0: 
                    temp = 0
                elif temp > 100:
                    temp = 100
                df_weather_copy.loc[i, "cal_rh"] = temp
            else:
                rt_tt2 = df_weather_copy.loc[i, "rt_tt2"]
                rt_dt = df_weather_copy.loc[i, "rt_dt"]
                logger.info(f"rt_tt2 is {rt_tt2}, rt_dt is {rt_dt}")
        # 特征筛选
        weather_features = [
            "rt_ssr",   # 太阳总辐射
            "rt_ws10",  # 10m 风速
            "rt_tt2",   # 2M 气温
            "cal_rh",   # 相对湿度
            "rt_ps",    # 气压
            "rt_rain",  # 降雨量
        ]
        df_weather_copy = df_weather_copy[["ds"] + weather_features]
        # 合并目标数据和天气数据
        df_history_with_weather = pd.merge(df_history_with_weather, df_weather_copy, on="ds", how="left") 
        # 插值填充缺失值
        df_history_with_weather = df_history_with_weather.interpolate()
        df_history_with_weather.dropna(inplace=True, ignore_index=True)
        return df_history_with_weather, weather_features


def extend_future_weather_features(df_future: pd.DataFrame, df_weather_future: pd.DataFrame=None):
    """
    未来天气数据特征构造
    """
    if df_weather_future is None:
        return df_future, []
    else:
        # df_future copy
        df_future_with_weather = df_future.copy()
        df_weather_future_copy = df_weather_future.copy()
        # 筛选天气预测数据
        pred_weather_features = ["pred_ssrd", "pred_ws10", "pred_tt2", "pred_rh", "pred_ps", "pred_rain"] 
        df_weather_future_copy = df_weather_future_copy[["ds"] + pred_weather_features]
        # 删除含空值的行
        df_weather_future_copy.dropna(inplace=True, ignore_index=True)
        # 数据类型转换
        for col in pred_weather_features:
            df_weather_future_copy[col] = df_weather_future_copy[col].apply(lambda x: float(x))
        # 将预测天气数据整理到预测df中
        df_future_with_weather["rt_ssr"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_ssrd"])
        df_future_with_weather["rt_ws10"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_ws10"])
        df_future_with_weather["rt_tt2"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_tt2"])
        df_future_with_weather["cal_rh"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_rh"])
        df_weather_future_copy["pred_ps"] = df_weather_future_copy["pred_ps"].apply(lambda x: x - 50.0)
        df_future_with_weather["rt_ps"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_ps"])
        df_weather_future_copy["pred_rain"] = df_weather_future_copy["pred_rain"].apply(lambda x: x - 2.5)
        df_future_with_weather["rt_rain"] = df_future_with_weather["ds"].map(df_weather_future_copy.set_index("ds")["pred_rain"])
        # features
        weather_features = [
            "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"
        ]
        return df_future_with_weather, weather_features


# def extend_lag_features(df: pd.DataFrame, 
#                         target: str, 
#                         group_col: str = None, 
#                         numLags: int = 3, 
#                         numHorizon: int = 0, 
#                         dropna: bool = False):
#     """
#     Time delay embedding.
#     Time series for supervised learning.

#     Args:
#         target (str): _description_
#         group_col (str, optional): _description_. Defaults to None.
#         numLags (int, optional): number of past values to used as explanatory variables.. Defaults to 1.
#         numHorizon (int, optional): how many values to forecast. Defaults to 0.
#         dropna (bool, optional): _description_. Defaults to False.
#     """
#     # 滞后特征构造
#     df_with_lags = df.copy()
#     # for i in range(1, self.numLags + 1):
#     for i in range(numLags, -numHorizon, -1):
#         if group_col is None:
#             if i <= 0:
#                 df_with_lags[f"{target}(t+{abs(i)+1})"] = df_with_lags[target].shift(i)
#             else:
#                 df_with_lags[f"{target}(t-{numLags + 1 - i})"] = df_with_lags[target].shift(i)
#         else:
#             if i <= 0:
#                 df_with_lags[f"{target}(t+{abs(i)+1})"] = df_with_lags.groupby(group_col)[target].shift(i)
#             else:
#                 df_with_lags[f"{target}(t-{numLags + 1 - i})"] = df_with_lags.groupby(group_col)[target].shift(i)
#     # 缺失值处理
#     if dropna:
#         df_with_lags = df_with_lags.dropna()
#         df_with_lags = df_with_lags.reset_index(drop = True)
    
#     # 滞后特征
#     lag_features = [
#         col for col in df_with_lags 
#         if col.startswith(f"{target}(")
#     ]
    
#     return df_with_lags, lag_features


def extend_lag_features(df_history: pd.DataFrame, target: str, lags: List):
    """
    添加滞后特征
    """
    if lags == []:
        return df_history, lags
    else:
        # df_history copy
        df_history_with_lag = df_history.copy()
        # 滞后特征构造
        for lag in lags:
            df_history_with_lag[f'lag_{lag}'] = df_history_with_lag[target].shift(lag)
        # 删除缺失值
        df_history_with_lag.dropna(inplace=True)
        # 滞后特征
        lag_features = [f'lag_{lag}' for lag in lags]
        # 滞后特征数据处理
        for lag_feature in lag_features:
            df_history_with_lag[lag_feature] = df_history_with_lag[lag_feature].apply(
                lambda x: float(x)
            )
        return df_history_with_lag, lag_features




# 测试代码 main 函数
def main():
    import datetime

    # input info
    pred_method = "multip-step_directly"                                           # 预测方法
    freq = "1h"                                                                    # 数据频率
    lags = 0                                                                       # 滞后特征构建
    target = "load"                                                                # 预测目标变量名称
    n_windows = 1                                                                  # cross validation 窗口数量
    history_days = 14                                                              # 历史数据天数
    predict_days = 1                                                               # 预测未来1天的功率
    data_length = 8 * 24 if n_windows > 1 else history_days * 24                   # 训练数据长度
    horizon = predict_days * 24                                                    # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
    now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
    start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
    future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻

    # data
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-17 00:00:00", end="2024-11-17 09:00:00", freq="1h"),
        "unique_id": [1] * 10,
        "load": range(1, 11),
        # "load2": np.random.randn(100),
    }) 
    logger.info(f"df: \n{df}")

    # ------------------------------
    # extend_lag_features test
    # ------------------------------
    df_with_lag, lag_features = extend_lag_features(df_history=df, target = "load", lags = [1, 2, 5])
    logger.info(f"df_with_lag: \n{df_with_lag} \nlag_features: {lag_features}")

    df_with_lag, lag_features = extend_lag_features(df_history=df, target = "load", lags = [])
    logger.info(f"df_with_lag: \n{df_with_lag} \nlag_features: {lag_features}")
    
    """
    # ------------------------------
    # cv_plot_df_window test
    # ------------------------------
    cv_plot_df_window = pd.DataFrame()
    cv_timestamp_df = pd.DataFrame({
        "ds": pd.date_range(start=start_time, end=now_time, freq=freq, inclusive="left")
    })
    cv_plot_df_window["ds"] = cv_timestamp_df[-24:]
    logger.info(cv_plot_df_window)
    """

    # ------------------------------
    # extend_datetime_features test
    # ------------------------------  
    df1, datetime_features = extend_datetime_features(df, feature_names = ["day", "hour"])
    logger.info(f"df1: \n{df1} \ndatetime_features: {datetime_features}")
    
    df2, datetime_features = extend_datetime_features(df, feature_names = [])
    logger.info(f"df2: \n{df2} \ndatetime_features: {datetime_features}")

    df_future = pd.DataFrame({
        "ds": pd.date_range(start=now_time, end=future_time, freq=freq, inclusive="left")
    })
    df3, datetime_features = extend_datetime_features(df_future, feature_names = [
        'minute', 'hour', 'day', 
        'weekday', 'week', 'day_of_week', 'week_of_year', 
        'month', 'days_in_month', 'quarter', 
        'day_of_year', 'year'
    ])
    logger.info(f"df3: \n{df3} \ndatetime_features: {datetime_features}")

if __name__ == "__main__":
    main()
