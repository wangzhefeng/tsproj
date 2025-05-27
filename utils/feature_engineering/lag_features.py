# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lag_features.py
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
from typing import List

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    # extend_lag_features test
    # ------------------------------
    df_with_lag, lag_features = extend_lag_features(df_history=df, target = "load", lags = [1, 2, 5])
    logger.info(f"df_with_lag: \n{df_with_lag} \nlag_features: {lag_features}")

    df_with_lag, lag_features = extend_lag_features(df_history=df, target = "load", lags = [])
    logger.info(f"df_with_lag: \n{df_with_lag} \nlag_features: {lag_features}")

if __name__ == "__main__":
    main()
