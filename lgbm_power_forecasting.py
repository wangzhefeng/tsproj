# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgbm_power_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-11
# * Version     : 1.0.021110
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
import copy
import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from models.LightGBM_forecast import Model
from utils.feature_engine import (
    extend_datetime_stamp_feature, 
    extend_date_type_feature, 
    extend_lag_feature
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_data(data_dir: str, data_cfgs: Dict[str, Any] = {}):
    """
    数据加载

    Args:
        project (str): _description_
        data_cfgs (Dict[str, Any], optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    # 时间索引
    start_date_str = data_cfgs["time_range"]["start_time"].strftime('%Y%m%d')
    now_date_start_str = data_cfgs["time_range"]["now_time_start"].strftime("%Y%m%d")
    now_date_end_str = data_cfgs["time_range"]["now_time_end"].strftime("%Y%m%d")
    future_date_str = data_cfgs["time_range"]["future_time"].strftime('%Y%m%d')
    # 时间范围字符串
    ts_range_history = f"{start_date_str}_to_{now_date_start_str}"
    ts_range_future = f"{now_date_end_str}_to_{future_date_str}" 
    # 历史数据
    df_load_1_history = pd.read_csv(os.path.join(data_dir, f"df_gate_{ts_range_history}.csv"), encoding="utf-8")
    df_load_2_history = pd.read_csv(os.path.join(data_dir, f"df_es_1_{ts_range_history}.csv"), encoding="utf-8")
    df_date_history = pd.read_csv(os.path.join(data_dir, f"df_date_history_{ts_range_history}.csv"), encoding="utf-8")
    df_weather_history = pd.read_csv(os.path.join(data_dir, f"df_weather_history_{ts_range_history}.csv"), encoding="utf-8")
    logger.info(f"df_load_1_history: \n{df_load_1_history}, \ndf_load_1_history.shape: {df_load_1_history.shape}")
    logger.info(f"df_load_2_history: \n{df_load_2_history}, \ndf_load_2_history.shape: {df_load_2_history.shape}")
    logger.info(f"df_date_history: \n{df_date_history}, \ndf_date_history.shape: {df_date_history.shape}")
    logger.info(f"df_weather_history: \n{df_weather_history}, \ndf_weather_history.shape: {df_weather_history.shape}")
    # 未来数据
    df_date_future = pd.read_csv(os.path.join(data_dir, f"df_date_future_{ts_range_future}.csv"), encoding="utf-8")
    df_weather_future = pd.read_csv(os.path.join(data_dir, f"df_weather_future_{ts_range_future}.csv"), encoding="utf-8")
    logger.info(f"df_date_future: \n{df_date_future}, \ndf_date_future.shape: {df_date_future.shape}")
    logger.info(f"df_weather_future: \n{df_weather_future}, \ndf_weather_future.shape: {df_weather_future.shape}")
    # 输入数据以字典形式整理
    input_data = {
        "df_load_1": df_load_1_history,
        "df_load_2": df_load_2_history,
        "df_date": df_date_history,
        "df_weather": df_weather_history,
        "df_date_future": df_date_future,
        "df_weather_future": df_weather_future,
    }
    
    return input_data


def process_history_data(input_data: Dict, data_cfgs: Dict):
    """
    历史数据预处理
    """
    # 时间特征处理
    def datetime_process(data, old_ds_name: str, new_ds_name: str = "ds"):
        df = copy.deepcopy(data)
        df[new_ds_name] = pd.to_datetime(df[old_ds_name])
        df.drop_duplicates(subset=new_ds_name, keep="last", inplace=True, ignore_index=True)
        return df
    df_load_1 = datetime_process(input_data["df_load_1"], "count_data_time", "ds")
    df_load_2 = datetime_process(input_data["df_load_2"], "count_data_time", "ds")
    df_weather_history = datetime_process(input_data["df_weather"], "ts", "ds")
    df_date_history = datetime_process(input_data["df_date"], "date", "ds")

    # 处理其他特征
    df_history = pd.DataFrame({"ds": pd.date_range(
        data_cfgs["time_range"]["start_time"], 
        data_cfgs["time_range"]["now_time_start"], 
        freq=data_cfgs["freq"]
    )})
    
    # y: load
    df_history["load_1"] = df_history["ds"].map(df_load_1.set_index("ds")["h_total_use"])
    df_history["load_1"] = df_history["load_1"].apply(lambda x: float(x))
    df_history["load_2"] = df_history["ds"].map(df_load_2.set_index("ds")["h_total_use"])
    df_history["load_2"] = df_history["load_2"].apply(lambda x: float(x))
    df_history["load"] = df_history["load_1"] + df_history["load_2"]
    df_history = df_history[df_history["load"] > data_cfgs["demand_load_min_thread"]]
    logger.info(f"df_history: \n{df_history.head()} \ndf_history.shape: {df_history.shape}")
    
    # weather data feature
    df_weather_history = df_weather_history[["ds", "rt_ssr", "rt_ws10", "rt_tt2", "rt_dt", "rt_ps", "rt_rain"]]
    for col in df_weather_history.columns[1:]:
        df_weather_history[col] = df_weather_history[col].apply(lambda x: float(x))
    # 通过温度和露点温度计算相对湿度
    df_weather_history["cal_rh"] = np.nan
    for i in df_weather_history.index:
        if (df_weather_history.loc[i, "rt_tt2"] is not np.nan and df_weather_history.loc[i, "rt_dt"] is not np.nan):
            temp = (math.exp(17.2693 * (df_weather_history.loc[i, "rt_dt"] - 273.15) / (df_weather_history.loc[i, "rt_dt"] - 35.86)) 
                  / math.exp(17.2693 * (df_weather_history.loc[i, "rt_tt2"] - 273.15) / (df_weather_history.loc[i, "rt_tt2"] - 35.86)) * 100)
            temp = 100 if temp > 100 else 0
            df_weather_history.loc[i, "cal_rh"] = temp
        else:
            rt_tt2 = df_weather_history.loc[i, "rt_tt2"]
            rt_dt = df_weather_history.loc[i, "rt_dt"]
            logger.info(f"rt_tt2: {rt_tt2}, rt_dt: {rt_dt}")
    logger.info(f"df_weather_history: \n{df_weather_history.head()} \ndf_weather_history.shape: {df_weather_history.shape}")
    # merge load and weather data
    df_history = pd.merge(df_history, df_weather_history, on = "ds", how = "left")
    logger.info(f"df_history: \n{df_history.head()} \ndf_history.shape: {df_history.shape}")

    # TODO 测试是否填充(ffill, bfill)对模型的影响 缺失值填充
    df_history = df_history.interpolate()
    df_history = df_history.ffill()
    df_history = df_history.bfill()
    logger.info(f"df_history: \n{df_history.head()} \ndf_history.shape: {df_history.shape}")
    # 删除含有缺失值的行
    df_history.dropna(inplace=True, ignore_index=True)
    logger.info(f"df_history: \n{df_history.head()} \ndf_history.shape: {df_history.shape}")

    return df_history, df_date_history


def process_future_data(input_data: Dict, data_cfgs: Dict):
    """
    未来数据预处理
    """
    # 时间特征处理
    def datetime_process(data, old_ds_name: str, new_ds_name: str = "ds"):
        df = copy.deepcopy(data)
        df[new_ds_name] = pd.to_datetime(df[old_ds_name])
        df.drop_duplicates(subset=new_ds_name, keep="last", inplace=True, ignore_index=True)
        return df
    df_weather_future = datetime_process(input_data["df_weather_future"], "ts", "ds")
    df_date_future = datetime_process(input_data["df_date_future"], "date", "ds")

    # 处理其他特征
    df_future = pd.DataFrame({"ds": pd.date_range(
        data_cfgs["time_range"]["now_time_end"], 
        data_cfgs["time_range"]["future_time"], 
        freq=data_cfgs["freq"]
    )})
    logger.info(f"df_future: \n{df_future.head()} \ndf_future.shape: {df_future.shape}")
    
    # 气象数据预处理
    df_weather_future = df_weather_future[["ds", "pred_ssrd", "pred_ws10", "pred_tt2", "pred_rh", "pred_ps", "pred_rain"]]
    for col in df_weather_future.columns[1:]:
        df_weather_future[col] = df_weather_future[col].apply(lambda x: float(x))
    logger.info(f"df_weather_future: \n{df_weather_future.head()} \ndf_weather_future.shape: {df_weather_future.shape}")
    # 将预测天气数据整理到预测df中
    df_future["rt_ssr"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ssrd"])
    df_future["rt_ws10"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ws10"])
    df_future["rt_tt2"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_tt2"])
    df_future["cal_rh"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_rh"])
    df_future["rt_ps"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_ps"])
    df_future["rt_rain"] = df_future["ds"].map(df_weather_future.set_index("ds")["pred_rain"])
    logger.info(f"df_future: \n{df_future.head()} \ndf_future.shape: {df_future.shape}")

    # TODO 测试是否填充(ffill, bfill)对模型的影响 缺失值填充
    df_future = df_future.interpolate()
    df_future = df_future.ffill()
    df_future = df_future.bfill()
    logger.info(f"df_future: \n{df_future.head()} \ndf_future.shape: {df_future.shape}")
    # 删除含有缺失值的行
    df_future.dropna(inplace=True, ignore_index=True)
    logger.info(f"df_future: \n{df_future.head()} \ndf_future.shape: {df_future.shape}")
    
    return df_future, df_date_future


def feature_engine_history_v1(df_history: pd.DataFrame, df_date_history: Dict):
    """
    # 时间戳特征构造
    df_history = extend_datetime_stamp_feature(
        df = df_history,
        feature_names=[
            "minute", "hour", "day", "weekday", "week", 
            "day_of_week", "week_of_year", "month", "days_in_month", 
            "quarter", "day_of_year", "year"
        ]
    )
    """
    # 日期类型特征构造
    df_history = extend_date_type_feature(df_history, df_date_history)
    # 特征筛选
    predict_features = [
        "ds",
        "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain",
        "date_type", 
        "load",
    ]
    df_history = df_history[predict_features]
    # 缺失值删除
    df_history.dropna(inplace=True, ignore_index=True)
    logger.info(f"df_history: \n{df_history.head()} \ndf_history.shape: {df_history.shape} \ndf_history.columns: {df_history.columns}")
    """
    # 特征筛选
    predict_features = [
        "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain",
        "datetime_minute", "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week", "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month", "datetime_quarter", "datetime_day_of_year", "datetime_year",
        "date_type",
    ]
    target_feature = "load"
    # 工作日数据（工作日和非节日的周六）
    df_history_workday = copy.deepcopy(df_history.query("(date_type == 1) or ((date_type == 2) and (datetime_weekday == 5))"))  # 工作日或者非节日的周六
    data_X_workday = df_history_workday[predict_features]
    data_Y_workday = df_history_workday[target_feature]
    logger.info(f"data_X_workday: \n{data_X_workday.head()} \ndata_X_workday.shape: {data_X_workday.shape} \ndata_X_workday.columns: {data_X_workday.columns}")
    logger.info(f"data_Y_workday: \n{data_Y_workday.head()} \ndata_Y_workday.shape: {data_Y_workday.shape}")
    # 非工作日数据（节日和非节日的周日）
    df_history_offday = copy.deepcopy(df_history.query("(date_type > 2) or ((date_type == 2) and (datetime_weekday == 6))"))  # 节日或者非节日的周日
    data_X_offday = df_history_offday[predict_features]
    data_Y_offday = df_history_offday[target_feature]
    logger.info(f"data_X_offday: \n{data_X_offday.head()} \ndata_X_offday.shape: {data_X_offday.shape} \ndata_X_offday.columns: {data_X_offday.columns}")
    logger.info(f"data_Y_offday: \n{data_Y_offday.head()} \ndata_Y_offday.shape: {data_Y_offday.shape}")

    return (
        data_X_workday,
        data_Y_workday,
        data_X_offday,
        data_Y_offday,
    )
    """
    return df_history


def feature_engine_future_v1(df_future: pd.DataFrame, df_date_future: pd.DataFrame):
    """
    # 时间戳特征构造
    df_future = extend_datetime_stamp_feature(
        df = df_future,
        feature_names=[
            "minute", "hour", "day", "weekday", "week", 
            "day_of_week", "week_of_year", "month", "days_in_month", 
            "quarter", "day_of_year", "year"
        ]
    )
    """
    # 日期类型特征构造
    df_future = extend_date_type_feature(df_future, df_date_future)
    # 特征筛选
    predict_features = [
        "ds",
        "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain",
        "date_type",
    ]
    df_future = df_future[predict_features]
    # 缺失值删除
    df_future.dropna(inplace=True, ignore_index=True)
    logger.info(f"df_future: \n{df_future.head()} \ndf_future.shape: {df_future.shape} \ndf_future.columns: {df_future.columns}")
    """
    # 特征筛选
    predict_features = [
        "rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain",
        "datetime_minute", "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week", "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month", "datetime_quarter", "datetime_day_of_year", "datetime_year",
        "date_type",
    ]
    # 工作日模型预测
    df_future_workday = copy.deepcopy(df_future.query("(date_type == 1) or ((date_type == 2) and (datetime_weekday == 5))"))  # 工作日或者非节日的周六
    X_future_workday = df_future_workday[predict_features]
    logger.info(f"X_future_workday: \n{X_future_workday.head()} \nX_future_workday.shape: {X_future_workday.shape} \nX_future_workday.columns: {X_future_workday.columns}")
    # 非工作日模型预测
    df_future_offday = copy.deepcopy(df_future.query("(date_type > 2) or ((date_type == 2) and (datetime_weekday == 6))"))  # 节日或者非节日的周日
    X_future_offday = df_future_offday[predict_features]
    logger.info(f"X_future_offday: \n{X_future_offday.head()} \nX_future_offday.shape: {X_future_offday.shape} \nX_future_offday.columns: {X_future_offday.columns}")

    return (
        X_future_workday,
        X_future_offday
    )
    """
    return df_future


def training_predicting(model_cfgs: Dict, history_data: pd.DataFrame, future_data: pd.DataFrame):
    """
    模型训练、预测

    Args:
        history_data (_type_): history data
        future_data (_type_): future data
        is_workday (str): 是否为工作日（根据具体的业务确定）
    """
    # model
    model = Model(
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    ) 
    # model running
    pred_df, eval_scores_df, cv_plot_df = model.run()
    # with pd.option_context("display.max_columns", None, "display.max_rows", None):
    logger.info(f"pred_df: \n{pred_df}")
    logger.info(f"eval_scores_df: \n{eval_scores_df}")
    logger.info(f"cv_plot_df: \n{cv_plot_df.sort_values(by="ds")}")





# 测试代码 main 函数
def main():
    logger.info("=" * 50)
    logger.info("Load parameters for traning...")
    logger.info("=" * 50)
    # project params
    project = "ashichuang"
    node = "asc2"
    # data params
    now = datetime.datetime(2025, 2, 6, 23, 46, 0)  # 模型预测的日期时间
    history_days = 30  # 历史数据天数
    predict_days = 1  # 预测未来1天的功率
    freq = "15min"  # 数据频率
    target = "load"  # 预测目标变量名称
    now_time_start = now.replace(tzinfo=None, minute=(now.minute // 15) * 15, second=0, microsecond=0)  # 历史数据结束时刻
    start_time = now_time_start.replace(hour=0, minute=0, second=0) - datetime.timedelta(days=history_days - 1)  # 历史数据开始时刻 
    now_time_end = now_time_start + datetime.timedelta(minutes=15)  # 未来数据开始时刻
    future_time = now_time_start + datetime.timedelta(days=predict_days)  # 未来数据结束时刻
    is_workday = True
    # training params
    is_training = True  # 是否进行模型训练
    is_predicting = True  # 是否进行模型预测
    # pred_method = "multip-step-directly"  # 预测方法
    pred_method = "multip-step-recursion"  # 预测方法
    lags = 0 if pred_method == "multip-step-directly" else 96  # 滞后特征构建
    n_windows = 15  # cross validation 窗口数量
    data_length = 15 * 96 if n_windows > 1 else history_days * 24 * 4  # 训练数据长度
    # data_length = 285
    horizon = predict_days * 24 * 4  # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    # input data path
    data_path = f"./dataset/{project}_dev_{now_time_start.strftime('%Y%m%d')}_hist{history_days}days_pred{predict_days}days/{node}/pred/" 
    # result save path
    result_path = f"./saved_results/predict_results/{project}/{node}_dev_{now_time_start.strftime('%Y%m%d')}_hist{history_days}days_pred{predict_days}days/" 
    os.makedirs(result_path, exist_ok=True)
    # model params
    model_cfgs = {
        "project": project,
        "node": node,
        "nodes": {
            "node": {
                "node_id": "f7a388e48987a8003245d4c7028fed70",
                "out_system_id": "nengyuanzongbiao",
                "in_system_id": "",
                "node_name": "阿石创新材料公司储能组2",
            },
        },
        "history_days": history_days,
        "predict_days": predict_days,
        "freq": freq,
        "target": target,
        "data_path": data_path,
        "time_range": {
            "start_time": start_time,
            "now_time_start": now_time_start,
            "now_time_end": now_time_end,
            "future_time": future_time,
        },
        "is_workday": is_workday,
        "is_training": is_training,
        "is_predicting": is_predicting,
        "pred_method": pred_method, 
        "lags": lags,
        "n_windows": n_windows,
        "data_length": data_length,
        "horizon": horizon, 
        "demand_load_min_thread": 66,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
        "result_path": result_path,
    }
    logger.info(f"Project: {project}")
    logger.info(f"Node: {node}")
    logger.info(f"history data days: {history_days}")
    logger.info(f"predict data days: {predict_days}")
    logger.info(f"freq: {freq}")
    logger.info(f"target: {target}")
    logger.info(f"history start time: {start_time}")
    logger.info(f"history end time: {now_time_start}")
    logger.info(f"future time start: {now_time_end}")
    logger.info(f"future time end: {future_time}")
    logger.info(f"is training: {is_training}")
    logger.info(f"is_predicting: {is_predicting}")
    logger.info(f"predict method: {pred_method}")
    logger.info(f"lags: {lags}")
    logger.info(f"n_windows: {n_windows}")
    logger.info(f"data_length: {data_length}")
    logger.info(f"horizon: {horizon}")
    logger.info(f"Input data path: {data_path}")
    logger.info(f"Result save path: {result_path}")
    # ------------------------------
    # 1.load data
    # ------------------------------
    logger.info("=" * 50)
    logger.info(f"Loading history and future data for training...")
    logger.info("=" * 50)
    # load raw data
    input_data = load_data(data_dir=data_path, data_cfgs=model_cfgs)
    # ------------------------------
    # data preprocessing
    # ------------------------------
    logger.info("=" * 50)
    logger.info(f"Processing history and future data for training...")
    logger.info("=" * 50)
    logger.info(f"Processing history data for training...")
    logger.info("-" * 40)
    data_history, data_date_history = process_history_data(input_data=input_data, data_cfgs=model_cfgs)
    logger.info(f"Processing future data for training...")
    logger.info("-" * 40)
    data_future, data_date_future = process_future_data(input_data=input_data, data_cfgs=model_cfgs)
    # ------------------------------
    # feature engine
    # ------------------------------
    logger.info("=" * 50)
    logger.info(f"History and Future data feature engine for training...")
    logger.info("=" * 50)
    logger.info(f"History data feature engine for training...")
    logger.info("-" * 40)
    df_history = feature_engine_history_v1(data_history, data_date_history)
    df_history_path = os.path.join(data_path, "df_history.csv")
    if not os.path.exists(df_history_path):
        df_history.to_csv(df_history_path)
        logger.info(f"df_history has saved in {df_history_path}")
    logger.info(f"Future data feature engine for training...")
    logger.info("-" * 40)
    df_future= feature_engine_future_v1(data_future, data_date_future) 
    df_future_path = os.path.join(data_path, "df_future.csv")
    if not os.path.exists(df_future_path):
        df_future.to_csv(df_future_path)
        logger.info(f"df_future has saved in {df_future_path}")
    # ------------------------------
    # model training and prdicting
    # ------------------------------
    logger.info("=" * 50)
    logger.info(f"Model training and predict...")
    logger.info("=" * 50)
    logger.info(f"Model training and multip-step-directly predict...")
    training_predicting(model_cfgs, df_history, df_future)

    # ------------------------------
    # logger.info(f"Model training and multip-step-recursion predict...")
    # training_predicting(model_cfgs, df_history, df_future)

if __name__ == "__main__":
    main()
