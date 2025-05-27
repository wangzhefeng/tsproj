# -*- coding: utf-8 -*-

# ***************************************************
# * File        : weather_featurs.py
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
import math

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
