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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO
# def extend_datetime_stamp_feature(df: pd.DataFrame):
#     """
#     增加时间特征
#     """    
#     df["datetime_minute"] = df["ds"].apply(lambda x: x.minute)
#     df["datetime_hour"] = df["ds"].apply(lambda x: x.hour)
#     df["datetime_day"] = df["ds"].apply(lambda x: x.day)

#     df["datetime_weekday"] = df["ds"].apply(lambda x: x.weekday())
#     df["datetime_week"] = df["ds"].apply(lambda x: x.week)
#     df["datetime_day_of_week"] = df["ds"].apply(lambda x: x.dayofweek)

#     df["datetime_week_of_year"] = df["ds"].apply(lambda x: x.weekofyear)
#     df["datetime_month"] = df["ds"].apply(lambda x: x.month)
#     df["datetime_days_in_month"] = df["ds"].apply(lambda x: x.daysinmonth)

#     df["datetime_quarter"] = df["ds"].apply(lambda x: x.quarter)
#     df["datetime_day_of_year"] = df["ds"].apply(lambda x: x.dayofyear)
#     df["datetime_year"] = df["ds"].apply(lambda x: x.year)

#     return df


# TODO
def extend_datetime_stamp_feature(df: pd.DataFrame, feature_names: str):
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
        df[f"datetime_{feature_name}"] = df["ds"].apply(func)

    return df


def extend_date_type_feature(df: pd.DataFrame, df_date: pd.DataFrame):
    """
    增加日期类型特征：
    1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
    """
    df["date"] = df["ds"].apply(
        lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
    )
    df["date_type"] = df["date"].map(df_date.set_index("date")["date_type"])

    return df


def extend_lag_feature(df: pd.DataFrame, target: str, group_col: str = None, numLags: int = 3, numHorizon: int = 0, dropna: bool = False):
    """
    Time delay embedding.
    Time series for supervised learning.

    Args:
        target (str): _description_
        group_col (str, optional): _description_. Defaults to None.
        numLags (int, optional): number of past values to used as explanatory variables.. Defaults to 1.
        numHorizon (int, optional): how many values to forecast. Defaults to 0.
        dropna (bool, optional): _description_. Defaults to False.
    """
    # 滞后特征构造
    tmp = df.copy()
    # for i in range(1, self.numLags + 1):
    for i in range(numLags, -numHorizon, -1):
        if group_col is None:
            if i <= 0:
                tmp[f"{target}(t+{abs(i)+1})"] = tmp[target].shift(i)
            else:
                tmp[f"{target}(t-{numLags + 1 - i})"] = tmp[target].shift(i)
        else:
            if i <= 0:
                tmp[f"{target}(t+{abs(i)+1})"] = tmp.groupby(group_col)[target].shift(i)
            else:
                tmp[f"{target}(t-{numLags + 1 - i})"] = tmp.groupby(group_col)[target].shift(i)
    # 缺失值处理
    if dropna:
        tmp = tmp.dropna()
        tmp = tmp.reset_index(drop = True)
    
    return tmp




# 测试代码 main 函数
def main():
    # data
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-11-17 00:00:00", end="2024-11-17 09:00:00", freq="1h"),
        "unique_id": [1] * 10,
        "load": range(1, 11),
        # "load2": np.random.randn(100),
    })
    
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
    
    cv_plot_df_window = pd.DataFrame()
    cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(start=start_time, end=now_time, freq=freq, inclusive="left")})
    cv_plot_df_window["ds"] = cv_timestamp_df[-24:]
    logger.info(cv_plot_df_window)

    # df_future = pd.DataFrame({"ds": pd.date_range(start=now_time, end=future_time, freq=freq, inclusive="left")})
    # df_3 = extend_datetime_stamp_feature(df_future, feature_names = [
    #     'minute', 'hour', 'day', 
    #     'weekday', 'week', 'day_of_week', 'week_of_year', 
    #     'month', 'days_in_month', 'quarter', 
    #     'day_of_year', 'year'
    # ])
    # logger.info(df_3)
    
    # df1 = extend_datetime_stamp_feature(df, feature_names = ["day", "hour"])
    # logger.info(df1)
    
    # df2 = extend_datetime_stamp_feature(df, feature_names = [])
    # logger.info(df2)

if __name__ == "__main__":
    main()
