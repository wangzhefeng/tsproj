# -*- coding: utf-8 -*-

# ***************************************************
# * File        : datetime_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040517
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






# 测试代码 main 函数
def main():
    import datetime
    from utils.log_util import logger

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
