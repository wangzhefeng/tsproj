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
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import math
from typing import List

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int, return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning

    Args:
        series: time series as pd.Series
        n_lags: number of past values to used as explanatory variables
        horizon: how many values to forecast
        return_Xy: whether to return the lags split from future observations

    Return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)
    # series name
    if series.name is None:
        name = 'Series'
    else:
        name = series.name
    # create features
    n_lags_iter = list(range(n_lags, -horizon, -1))
    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    # features rename
    df.columns = [
        f'{name}(t-{j - 1})' if j > 0 else f'{name}(t+{np.abs(j) + 1})'
        for j in n_lags_iter
    ]
    # df.columns = [re.sub('t-0', 't', x) for x in df.columns]
    # 返回 pandas.Dataframe
    if not return_Xy:
        return df
    # future features
    is_future = df.columns.str.contains('\\+')
    # feature split
    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y


# TODO
def extend_lag_feature(df: pd.DataFrame, 
                       cols: List,
                       group_col: str = None, 
                       numLags: int = 3, 
                       numHorizon: int = 0, 
                       dropna: bool = False):
    """
    Time delay embedding.
    Time series for supervised learning.

    Args:
        numLags (int, optional): number of past values to used as explanatory variables. Defaults to 1.
        numHorizon (int, optional): how many values to forecast. Defaults to 0.
    """
    # 滞后特征构造
    df_with_lags = df.copy()
    # for i in range(1, self.numLags + 1):
    for i in range(numLags, -numHorizon, -1):
        if group_col is None:
            if i <= 0:
                for col in cols:
                    df_with_lags[f"{col}(t+{abs(i)+1})"] = df_with_lags[col].shift(i)
            else:
                for col in cols:
                    df_with_lags[f"{col}(t-{numLags + 1 - i})"] = df_with_lags[col].shift(i)
        else:
            if i <= 0:
                for col in cols:
                    df_with_lags[f"{col}(t+{abs(i)+1})"] = df_with_lags.groupby(group_col)[col].shift(i)
            else:
                for col in cols:
                    df_with_lags[f"{col}(t-{numLags + 1 - i})"] = df_with_lags.groupby(group_col)[col].shift(i)
    # 缺失值处理
    if dropna:
        df_with_lags = df_with_lags.dropna()
        df_with_lags = df_with_lags.reset_index(drop = True)
    # 滞后特征
    lag_features = []
    for col in cols:
        lag_features += [col_name for col_name in df_with_lags if col_name.startswith(f"{col}(")]
        del df_with_lags[col]

    return df_with_lags, lag_features


# TODO
def extend_lag_feature_univariate(df: pd.DataFrame, target: str, lags: List):
    """
    添加滞后特征
    """
    df_lags = df.copy()
    # lag features building
    for lag in lags:
        df_lags[f'{target}_{lag}'] = df_lags[target].shift(lag)
    df_lags.dropna(inplace=True)
    
    lag_features = [f'{target}_{lag}' for lag in lags] 

    return df_lags, lag_features 


def extend_lag_feature_multivariate(df, exogenous_features: List, target: str, n_lags: int):
    """
    添加滞后特征
    """
    # 将 date 作为索引
    df.set_index("timeStamp", inplace=True)
    # delay embedding: lagged features
    lagged_features_ds = []
    for col in exogenous_features + [target]:
        col_df = time_delay_embedding(series=df[col], n_lags=n_lags, horizon = 1)
        lagged_features_ds.append(col_df)
        df = df.drop(columns=[col])
    lagged_features_df = pd.concat(lagged_features_ds, axis=1).dropna()
    lagged_features_df = lagged_features_df.reset_index()
    # 滞后特征提取
    lag_features = [
        col for col in lagged_features_df.columns 
        if col.__contains__("(t-") or col.__contains__(r"(t)")
    ]
    # 目标特征提取
    target_features = [
        col for col in lagged_features_df.columns 
        if col.__contains__("(t+")
    ]
    # 数据合并
    df = df.reset_index()
    df = lagged_features_df.merge(df, on = "timeStamp", how = "left")
    # 特征分割
    # pred_vars_1 = lagged_features_df.columns.str.contains(r"\(t\-")
    # pred_vars_2 = lagged_features_df.columns.str.contains(r"\(t\)")
    # pred_vars_total = []
    # for pred1, pred2 in zip(pred_vars_1, pred_vars_2):
    #     if pred1 or pred2:
    #         pred_vars_total.append(True)
    #     else:
    #         pred_vars_total.append(False)
    # targ_vars = lagged_features_df.columns.str.contains(r"\(t\+") 
    # X = lagged_features_df.iloc[:, pred_vars_total]
    # y = lagged_features_df.iloc[:, targ_vars]
    # logger.info(f"X.columns: \n{X.columns} \nX.shape: {X.shape}")
    # logger.info(f"y.columns: \n{y.columns} \ny.shape: {y.shape}")
    
    return df, lag_features, target_features


def extend_datetime_feature(df: pd.DataFrame, feature_names: str):
    """
    增加时间特征
    """
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
        df[f"datetime_{feature_name}"] = df["timeStamp"].apply(func)
    datetime_features = [
        col for col in df.columns 
        if col.startswith("datetime")
    ]

    return df, datetime_features


def extend_date_type_feature(df: pd.DataFrame, df_date: pd.DataFrame):
    """
    增加日期类型特征：
    1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
    """
    # data map
    df["date"] = df["ds"].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    df["date_type"] = df["date"].map(df_date.set_index("date")["date_type"])
    # date features
    date_features = ["date_type"]

    return df, date_features


def extend_weather_feature(df: pd.DataFrame, df_weather: pd.DataFrame):
    """
    处理天气特征
    """
    weather_features_raw = ["rt_ssr", "rt_ws10", "rt_tt2", "rt_dt", "rt_ps", "rt_rain"]
    df_weather = df_weather[["ds"] + weather_features_raw]
    # 删除含空值的行
    df_weather.dropna(inplace=True, ignore_index=True)
    # 将除了timeStamp的列转为float类型
    for col in weather_features_raw:
        df_weather[col] = df_weather[col].apply(lambda x: float(x))
    # 计算相对湿度
    df_weather["cal_rh"] = np.nan
    for i in df_weather.index:
        if (df_weather.loc[i, "rt_tt2"] is not np.nan
            and df_weather.loc[i, "rt_dt"] is not np.nan):
            # 通过温度和露点温度计算相对湿度
            temp = (
                math.exp(17.2693
                    * (df_weather.loc[i, "rt_dt"] - 273.15)
                    / (df_weather.loc[i, "rt_dt"] - 35.86))
                / math.exp(17.2693
                    * (df_weather.loc[i, "rt_tt2"] - 273.15)
                    / (df_weather.loc[i, "rt_tt2"] - 35.86))
                * 100
            )
            if temp < 0: 
                temp = 0
            elif temp > 100:
                temp = 100
            df_weather.loc[i, "cal_rh"] = temp
        else:
            rt_tt2 = df_weather.loc[i, "rt_tt2"]
            rt_dt = df_weather.loc[i, "rt_dt"]
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
    df_weather = df_weather[["ds"] + weather_features]
    # 合并目标数据和天气数据
    df = pd.merge(df, df_weather, on="ds", how="left")
    # 插值填充缺失值
    df = df.interpolate()
    df.dropna(inplace=True, ignore_index=True)
    
    return df, weather_features


def extend_future_weather_feature(df_future: pd.DataFrame, df_weather_future: pd.DataFrame):
    """
    未来天气数据特征构造
    """
    # 筛选天气预测数据
    pred_weather_features = ["pred_ssrd", "pred_ws10", "pred_tt2", "pred_rh", "pred_ps", "pred_rain"] 
    df_weather_future = df_weather_future[["ds"] + pred_weather_features]
    # 删除含空值的行
    df_weather_future.dropna(inplace=True, ignore_index=True)
    # 数据类型转换
    for col in pred_weather_features:
        df_weather_future[col] = df_weather_future[col].apply(lambda x: float(x))
    # 将预测天气数据整理到预测df中
    df_future["rt_ssr"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ssrd"])
    df_future["rt_ws10"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ws10"])
    df_future["rt_tt2"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_tt2"])
    df_future["cal_rh"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_rh"])
    df_weather_future["pred_ps"] = df_weather_future["pred_ps"].apply(lambda x: x - 50.0)
    df_future["rt_ps"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ps"])
    df_weather_future["pred_rain"] = df_weather_future["pred_rain"].apply(lambda x: x - 2.5)
    df_future["rt_rain"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_rain"])
    # features
    weather_features = [
        "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"
    ]
    
    return df_future, weather_features




# 测试代码 main 函数
def main():  
    # input info
    import datetime
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
    # ------------------------------
    # lag features
    # ------------------------------
    # data
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-17 00:00:00", end="2024-11-17 09:00:00", freq="1h"),
        # "unique_id": [1] * 10,
        "load1": range(1, 11),
        "load2": range(2, 12),
    })
    # df = df.set_index("ds")
    logger.info(f"df: \n{df}")
    
    # time_delay_embedding test
    # -------------------------
    # df_with_lags = []
    # for col in df.columns:
    #     col_df = time_delay_embedding(df[col], n_lags=3, horizon=2, return_Xy=False)
    #     df_with_lags.append(col_df)
    # df_with_lags = pd.concat(df_with_lags, axis=1).dropna()
    # logger.info(f"df_with_lags: \n{df_with_lags}")
    # logger.info(f"df_with_lags.columns: \n{df_with_lags.columns}")
    
    # X_lags, y_lags = time_delay_embedding(df["load1"], n_lags=3, horizon=0, return_Xy=True)
    # logger.info(f"X_lags: \n{X_lags} \ny_lags: \n{y_lags}")
    
    # extend_lag_feature test
    # -------------------------
    # df_lags, lags_features = extend_lag_feature(
    #     df=df,
    #     cols=["load1", "load2"],
    #     group_col=None,
    #     numLags=3,
    #     numHorizon=2,
    #     dropna=True,
    # )
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info(f"df_lags: \n{df_lags}")
    #     logger.info(f"lag_features: {lags_features}")
    
    # extend_lag_feature_univariate
    # -------------------------
    df_lags, lag_features = extend_lag_feature_univariate(df=df, target="load1", lags=[])
    logger.info(f"df_lags: \n{df_lags} \nlag_features: {lag_features}")
    
    # extend_lag_feature_multivariate
    # -------------------------
    # logger.info(f"df: \n{df}")
    df_lags, lag_features, target_features = extend_lag_feature_multivariate(df=df, target="load1", exogenous_features=[], n_lags=0)
    with pd.option_context("display.max_columns", None):
        logger.info(f"df_lags: \n{df_lags} \nlag_features: {lag_features} \ntarget_features: {target_features}")
    """
    # ------------------------------
    # extend_datetime_feature
    # ------------------------------
    df_future = pd.DataFrame({"ds": pd.date_range(start=now_time, end=future_time, freq=freq, inclusive="left")}) 
    
    df1 = extend_datetime_feature(df, feature_names = ["day", "hour"])
    logger.info(df1)
    
    df2 = extend_datetime_feature(df, feature_names = [])
    logger.info(df2)

    df_3 = extend_datetime_feature(df_future, feature_names = [
        'minute', 'hour', 'day', 
        'weekday', 'week', 'day_of_week', 'week_of_year', 
        'month', 'days_in_month', 'quarter', 
        'day_of_year', 'year'
    ])
    logger.info(df_3)
    """

if __name__ == "__main__":
    main()
