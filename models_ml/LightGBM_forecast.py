# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LightGBM_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.增加 log;
# *               2.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import copy
import math
import random
import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.feature_engine import (
    extend_datetime_stamp_feature,
    extend_lag_feature,
    extend_date_type_feature,
    extend_date_type_features,
    extend_lag_features,
    extend_datetime_features,
    extend_weather_features,
    extend_future_weather_features,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    def __init__(self, args: Dict, history_data: pd.DataFrame, future_data: pd.DataFrame) -> None:
        self.args = args
        # data
        self.history_data = history_data
        self.future_data = future_data
        # datetime index
        self.train_start_time_str = self.args.train_start_time.strftime('%Y%m%d')
        self.train_end_time_str = self.args.train_end_time.strftime("%Y%m%d")
        self.forecast_start_time_str = self.args.forecast_start_time.strftime("%Y%m%d")
        self.forecast_end_time_str = self.args.forecast_end_time.strftime('%Y%m%d')

    # TODO
    def __load_data(self):
        """
        数据加载
        """
        # 历史数据
        df_target = pd.read_csv(os.path.join(self.args.data_dir, f"df_target.csv"), encoding="utf-8")
        df_date_history = pd.read_csv(os.path.join(self.args.data_dir, f"df_date_history.csv"), encoding="utf-8")
        df_weather_history = pd.read_csv(os.path.join(self.args.data_dir, f"df_weather_history.csv"), encoding="utf-8")
        # 未来数据
        df_date_future = pd.read_csv(os.path.join(self.args.data_dir, f"df_date_future.csv"), encoding="utf-8")
        df_weather_future = pd.read_csv(os.path.join(self.args.data_dir, f"df_weather_future.csv"), encoding="utf-8")
        # 输入数据以字典形式整理
        input_data = {
            "df_target": df_target,
            "df_date_history": df_date_history,
            "df_weather_history": df_weather_history,
            "df_date_future": df_date_future,
            "df_weather_future": df_weather_future,
        }
        
        return input_data

    def __process(self, data: pd.DataFrame, date_col: str, new_date_col: str = "ds"):
        """
        时序数据预处理

        Args:
            data (pd.DataFrame): 时间序列数据
            date_col (str): 原时间戳列
            new_date_col (str): 新的时间戳列
        """
        # 数据拷贝
        df = copy.deepcopy(data)
        # 转换时间戳类型
        df[new_date_col] = pd.to_datetime(df[date_col])
        # 去除重复时间戳
        df.drop_duplicates(
            subset = new_date_col,
            keep = "last",
            inplace = True,
            ignore_index = True,
        )

        return df
    
    def __process_target(self, df_target: pd.DataFrame):
        """
        目标特征数据预处理
        """
        # 目标特征数据转换为浮点数
        df_target[self.args.target] = df_target[self.args.target].apply(lambda x: float(x))
        # 缺失值处理
        df_target.dropna(inplace=True, ignore_index=True)
        # 样本筛选
        df_target = df_target[df_target[self.args.target] > self.args.threshold]
        
        return df_target

    def __calc_features_corr(self, df_history, features):
        """
        分析预测特征与目标特征的相关性

        Args:
            df_history (_type_): _description_
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        # features correlation
        features_corr = df_history[features + ['load']].corr()
    
        return features_corr
    
    def __preprocessing_history_data(self, input_data: Dict):
        """
        历史数据预处理
        """
        # 数据预处理
        df_target = self.__process(input_data["df_target"], "count_data_time", "ds")
        df_target = self.__process_target(df_target)
        df_date_history = self.__process(input_data["df_date_history"], "date", "ds")
        df_weather_history = self.__process(input_data["df_weather_history"], "ts", "ds")

        # 生成以 freq 间隔的时间序列, 创建 DataFrame 并添加 timeStamp 列
        df_history = pd.DataFrame({"ds": pd.date_range(
            self.args.train_start_time, 
            self.args.train_end_time, 
            freq = self.args.freq
        )})
        # 将原始数据映射到时间戳完整的 df 中
        df_history["y"] = df_history["ds"].map(df_target.set_index("ds")[self.args.target]) 
        # 特征工程：滞后特征
        df_history, lag_features = extend_lag_features(df_history, lags = self.args.lags)
        # 特征工程：天气特征
        df_history, weather_features = extend_weather_features(df_history, df_weather_history)
        # 特征工程：日期时间特征
        df_history, datetime_features = extend_datetime_features(df_history)
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_history, date_features = extend_date_type_features(df_history, df_date_history)

        # 插值填充预测缺失值
        df_history = df_history.interpolate()
        df_history.dropna(inplace = True, ignore_index = True) 

        # 历史数据特征列表
        train_features = lag_features + \
            weather_features + \
            datetime_features + \
            date_features       
        # 特征排序
        df_history = df_history[["ds"] + train_features + [self.args.target]]

        # 归一化/标准化
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        if self.args.scale:
            df_history[train_features] = self.scaler_features.fit_transform(df_history[train_features])
            df_history[self.args.target] = self.scaler_target.fit_transform(df_history[[self.args.target]])
        
        # TODO 数据分割: 工作日预测特征，目标特征
        # df_history = copy.deepcopy(df_history[df_history["date_type"] == 1])

        return df_history

    def __preprocessing_future_data(self, input_data):
        """
        处理未来数据
        """
        # 数据预处理
        df_weather_future = self.__process(input_data["df_weather_future"], "ts", "ds")
        df_date_future = self.__process(input_data["df_date_future"], "date", "ds")

        # 生成以 freq 间隔的时间序列, 创建 DataFrame 并添加 timeStamp 列
        df_future = pd.DataFrame({"ds": pd.date_range(
            self.args.forecast_start_time, 
            self.args.forecast_end_time, 
            freq = self.args.freq
        )})
        
        # 特征工程: 天气特征
        df_future, weather_features = extend_future_weather_features(df_future, df_weather_future)
        # 特征工程: 日期时间特征
        df_future, datetime_features = extend_datetime_features(df_future)
        # 特征工程: 日期类型(节假日、特殊事件)特征
        df_future, date_features = extend_date_type_features(df_future, df_date_future)
        
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_future.dropna(inplace = True, ignore_index = True) 

        # 未来数据特征列表
        future_features = weather_features + datetime_features + date_features
        # 特征排序
        df_future = df_future[["ds"] + future_features]

        # 归一化/标准化
        if self.args.scale:
            df_future[future_features] = self.scaler_features.fit_transform(df_future[future_features])
            df_future[self.args.target] = self.scaler_target.fit_transform(df_future[[self.args.target]])

        # TODO 数据分割: 工作日预测特征, 目标特征
        # df_future = copy.deepcopy(df_future[df_future["date_type"] == 1])

        return df_future
 
    def training_v1(self, input_data: Dict):
        """
        模型训练、测试

        Args:
            data_X (_type_): _description_
            data_Y (_type_): _description_
            lgbm_params (_type_): _description_
            scaler_features (_type_): _description_
            scaler_target (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 历史数据预处理
        df_history = self.__preprocessing_history_data(input_data)
        data_X = df_history[df_history.columns[1:-1]]
        data_Y = df_history[self.args.target]
        
        # 训练集、测试集划分
        data_length = len(data_X)
        X_train = data_X.iloc[-data_length:-self.args.split_length]
        Y_train = data_Y.iloc[-data_length:-self.args.split_length]
        X_test = data_X.iloc[-self.args.split_length:]
        Y_test = data_Y.iloc[-self.args.split_length:]

        # 模型训练
        lgb_model = lgb.LGBMRegressor(**self.args.lgbm_params)
        lgb_model.fit(X_train, Y_train)
        # 特征重要性排序
        if self.args.plot_features_importance:
            lgb.plot_importance(
                lgb_model, 
                importance_type = "gain",  # gain, split
                figsize = (7, 6), 
                title = "LightGBM Feature Importance (Gain)"
            )
            plt.show()
        
        # 模型测试
        Y_pred = self.__recursive_forecast(
            model = lgb_model,
            history_df = pd.concat([X_train, Y_train], axis=1),
            future_df = X_test,
            train_features = X_train.columns,
            scaler_features = self.args.scaler_features,
            scaler_target = self.args.scaler_target,
        )
        # 模型评估
        test_scores = {
            "R2": r2_score(Y_test, Y_pred),
            "mse": mean_squared_error(Y_test, Y_pred),
            "rmse": root_mean_squared_error(Y_test, Y_pred),
            "mae": mean_absolute_error(Y_test, Y_pred),
            "mape": mean_absolute_percentage_error(Y_test, Y_pred),
            "accuracy": 1 - mean_absolute_percentage_error(Y_test, Y_pred), 
        }
        logger.info(f"R2: {test_scores['R2']:.4f}")
        logger.info(f"mse: {test_scores['mse']:.4f}")
        logger.info(f"rmse: {test_scores['rmse']:.4f}")
        logger.info(f"mape: {test_scores['mape']:.4f}")
        logger.info(f"mape: {test_scores['mape']:.4f}")
        logger.info(f"mape accuracy: {test_scores['accuracy']:.4f}")
        
        # 最终模型
        final_model = lgb.LGBMRegressor(**self.args.lgbm_params)
        final_model.fit(data_X, data_Y)

        return final_model, test_scores 
 
    def forecasting_v1(self, lgb_model, input_data):
        """
        时间序列预测

        Args:
            input_data (_type_): _description_
            lgb_model (_type_): _description_
            df_train (_type_): _description_
            train_features (_type_): _description_
            scaler_features (_type_): _description_
            scaler_target (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 历史数据预处理
        df_history = self.__preprocessing_history_data(input_data = input_data)
        # 未来数据预处理
        df_future = self.__preprocessing_future_data(input_data = input_data)
        # multi-step recursive forecast
        predictions = self.__recursive_forecast(
            model = lgb_model,
            history_df = df_history,
            future_df = df_future,
            train_features = df_history.columns[1:],
            scaler_features = self.scaler_features,
            scaler_target = self.scaler_target,
        )
        df_future[self.args.target] = predictions 
        # 输出结果处理
        df_future.dropna(inplace=True, ignore_index=True)

        return df_future

    def __process_output(self, df_future):
        """
        输出结果处理

        Args:
            df_future (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 特征处理
        for i in range(len(df_future)):
            df_future.loc[i, "id"] = df_future.loc[i, "ds"].strftime("%Y%m%d%H%M%S")
            df_future.loc[i, "predict_value"] = str(df_future.loc[i, self.args.target])
            df_future.loc[i, "predict_adjustable_amount"] = str(df_future.loc[i, self.args.target] * random.uniform(0.05, 0.1))
            df_future.loc[i, "timestamp"] = df_future.loc[i, "ds"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # 特征输出
        df_future = df_future[[
            "id",
            "predict_value",
            "predict_adjustable_amount",
            "timestamp",
        ]]

        return df_future
     
    # TODO --------------------------------------------------------------------- 
    def process_history_data(self, input_data: Dict, data_cfgs: Dict):
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
        df_history[self.args.target] = df_history["load_1"] + df_history["load_2"]
        df_history = df_history[df_history[self.args.target] > data_cfgs["demand_load_min_thread"]]
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

    def feature_engine_history_v1(self, df_history: pd.DataFrame, df_date_history: Dict):
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
            self.args.target,
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
        target_feature = self.args.target
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

    def feature_engine_future_v1(self, df_future: pd.DataFrame, df_date_future: pd.DataFrame):
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

    def training_predicting(self, args: Dict, history_data: pd.DataFrame, future_data: pd.DataFrame):
        """
        模型训练、预测

        Args:
            history_data (_type_): history data
            future_data (_type_): future data
            is_workday (str): 是否为工作日（根据具体的业务确定）
        """
        # model
        model = Model(
            args=args,
            history_data=history_data,
            future_data=future_data,
        ) 
        # model running
        pred_df, eval_scores_df, cv_plot_df = model.run()
        # with pd.option_context("display.max_columns", None, "display.max_rows", None):
        logger.info(f"pred_df: \n{pred_df}")
        logger.info(f"eval_scores_df: \n{eval_scores_df}")
        logger.info(f"cv_plot_df: \n{cv_plot_df.sort_values(by="ds")}")
    # TODO ---------------------------------------------------------------------
    
    def _get_history_data(self):
        """
        历史数据处理
        """
        # 数据时间范围
        start_time = self.args["time_range"]["start_time"]
        now_time = self.args["time_range"]["now_time_start"]
        freq = self.args["freq"]
        # 构造时间戳完整的历史数据：生成以 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_history = pd.DataFrame({"ds": pd.date_range(start=start_time, end=now_time, freq=freq)})
        # 复制 history data
        df = copy.deepcopy(self.history_data)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df["ds"] = pd.to_datetime(df["ds"])
            # 去除重复时间戳
            df.drop_duplicates(subset="ds", keep="last", inplace=True, ignore_index=True)
            # 数据处理
            for col in df.columns:
                if col != "ds":
                    # 将数据转换为浮点数类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, y, exogenous_features]
                    df_history[col] = df_history["ds"].map(df.set_index("ds")[col])
        
        return df_history

    def _get_future_data(self):
        """
        未来数据处理
        """
        # 数据时间范围
        now_time = self.args["time_range"]["now_time_end"]
        future_time = self.args["time_range"]["future_time"]
        freq = self.args["freq"]
        # 构造时间戳完整的历史数据：生成未来 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_future = pd.DataFrame({"ds": pd.date_range(start=now_time, end=future_time, freq=freq)})
        # 复制 history data
        df = copy.deepcopy(self.future_data)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df["ds"] = pd.to_datetime(df["ds"])
            # 去除重复时间戳
            df.drop_duplicates(subset="ds", keep="last", inplace=True, ignore_index=True)
            # 数据处理
            for col in df.columns:
                if col != "ds":
                    # 将数据转换为字符串类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, exogenous_features]
                    df_future[col] = df_future["ds"].map(df.set_index("ds")[col])
        
        return df_future

    @staticmethod
    def _get_datetime_features(df: pd.DataFrame, datetime_features: List = []):
        """
        日期/时间特征构造
        """
        df_with_datetime_features = extend_datetime_stamp_feature(df, datetime_features)
        datetime_feats = [col for col in df_with_datetime_features.columns if col.startswith("datetime")]
        
        return df_with_datetime_features, datetime_feats

    def _get_history_lag_features(self, df_history):
        """
        滞后特征构造
        """
        df_history_lags = extend_lag_feature(
            df=df_history, 
            target=self.args["target"], 
            group_col=None, 
            numLags=self.args["lags"],
            numHorizon=0, 
            dropna=True,
        )
        df_history_lags_feats = [
            col for col in df_history_lags 
            if col.startswith(f"{self.args['target']}(")
        ]
        
        return df_history_lags, df_history_lags_feats
    
    def _get_future_lag_features(self, df_history, df_future):
        """
        滞后特征构造
        """
        # params
        now_time = self.args["time_range"]["now_time_end"]
        future_time = self.args["time_range"]["future_time"]
        freq = self.args["freq"]
        # 特征构造
        df_future_lags = extend_lag_feature(
            df=df_history, 
            target=self.args["target"], 
            group_col=None, 
            numLags=0, 
            numHorizon=self.args["lags"], 
            dropna=False,
        )
        # 筛选样本
        df_future_lags = df_future_lags.iloc[-self.args["horizon"]:, ]
        # 时间戳修改为未来时间戳
        df_future_lags["ds"] = pd.date_range(start=now_time, end=future_time, freq=freq)
        # 滞后特征合并
        df_future_lags_feats = [col for col in df_future_lags if col.startswith(f"{self.args['target']}(")]
        for col in df_future_lags_feats:
            df_future[col] = df_future["ds"].map(df_future_lags.set_index("ds")[col])
        
        return df_future, df_future_lags_feats

    def process_input_history_data(self):
        """
        处理输入历史数据
        """
        # ------------------------------
        # TODO 数据预处理
        # ------------------------------
        # df_history = self._get_history_data()
        # logger.info(f"df_history: \n{df_history}")
        df_history = self.history_data
        # ------------------------------
        # 特征工程
        # ------------------------------
        # 目标特征
        # target_feats = [target]
        # 时间戳特征
        # timestamp_feats = ["ds"]
        # 外生特征
        exogenous_features = [col for col in df_history.columns if col != "ds" and col != self.args["target"]]
        # 日期时间特征
        df_history, datetime_features = self._get_datetime_features(
            df_history, 
            datetime_features = [
                # TODO
                'minute', 
                'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )
        # 滞后特征构造
        df_history, lag_features = self._get_history_lag_features(df_history)
        # ------------------------------
        # 缺失值处理
        # ------------------------------
        df_history = df_history.interpolate()  # 缺失值插值填充
        df_history.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        
        if self.args["is_workday"]:
            df_history = copy.deepcopy(df_history.query("(date_type == 1) or ((date_type == 2) and (datetime_weekday == 5))"))
            df_history_workday_path = os.path.join(self.args["data_path"], "df_history_workday.csv")
            if not os.path.exists(df_history_workday_path):
                df_history.to_csv(df_history_workday_path)
                logger.info(f"df_history_workday has saved in {df_history_workday_path}")
        else:
            df_history = copy.deepcopy(df_history.query("(date_type > 2) or ((date_type == 2) and (datetime_weekday == 6))"))
            df_history_offday_path = os.path.join(self.args["data_path"], "df_history_offday.csv")
            if not os.path.exists(df_history_offday_path):
                df_history.to_csv(df_history_offday_path)
                logger.info(f"df_history_offday has saved in {df_history_offday_path}")
        # ------------------------------
        # 预测特征、目标变量分割
        # ------------------------------
        # 特征筛选
        predict_features = datetime_features + exogenous_features + lag_features
        df_history_X = df_history[predict_features]
        df_history_Y = df_history[self.args["target"]]
        
        return df_history_X, df_history_Y

    def process_input_future_data(self):
        """
        处理输入未来数据
        """
        # ------------------------------
        # TODO 数据预处理
        # ------------------------------
        # df_history = self._get_history_data()
        # df_future = self._get_future_data()
        # logger.info(f"df_history: \n{df_history}")
        # logger.info(f"df_future: \n{df_future}")
        df_history = self.history_data
        df_future = self.future_data
        # ------------------------------
        # 缺失值处理
        # ------------------------------
        df_future = df_future.interpolate()  # 缺失值插值填充
        df_future.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        # ------------------------------
        # 特征工程
        # ------------------------------
        # 目标特征
        # target_feats = [target]
        # 时间戳特征
        # timestamp_feats = ["ds"]
        # 外生特征
        exogenous_features = [
            col 
            for col in df_future.columns 
            if col != "ds" and col != self.args["target"]
        ]
        # 日期时间特征
        df_future, datetime_features = self._get_datetime_features(
            df_future,
            datetime_features = [
                # TODO
                'minute', 
                'hour', 'day', 
                'weekday', 'week', 'day_of_week', 'week_of_year', 
                'month', 'days_in_month', 'quarter', 
                'day_of_year', 'year'
            ],
        )
        
        # TODO fixed
        if self.args["predict_days"] == 1:
            df_future_date_type = df_future["date_type"].unique()[0]
            df_future_datetime_weekday = df_future["datetime_weekday"].unique()[0]
            if df_future_date_type in [1, 2] or df_future_datetime_weekday == 5:
                self.args["is_workday"] = True
            elif df_future_date_type >= 2 or df_future_datetime_weekday == 6:
                self.args["is_workday"] = False
    
        # 滞后特征
        df_future, lag_features = self._get_future_lag_features(df_history, df_future)

        # 根据预测天数及待预测的天是否为工作日进行分别预测
        if self.args["predict_days"] > 1:
            if self.args["is_workday"]:
                df_future = copy.deepcopy(df_future.query("(date_type == 1) or ((date_type == 2) and (datetime_weekday == 5))"))
            else:
                df_future = copy.deepcopy(df_future.query("(date_type > 2) or ((date_type == 2) and (datetime_weekday == 6))"))
        else:
            df_future = df_future
        # ------------------------------
        # 预测特征数据
        # ------------------------------
        # 特征筛选
        predict_features = datetime_features + exogenous_features + lag_features
        df_future_X = df_future[predict_features]

        return df_future_X

    def _cv_split_index(self, window: int):
        """
        数据分割索引构建
        """
        valid_end   = -1        + (-self.args["horizon"]) * window
        valid_start = valid_end + (-self.args["horizon"]) + 1
        train_end   = valid_start
        train_start = valid_end + (-self.args["data_length"]) + 1

        return train_start, train_end, valid_start, valid_end

    def cv_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, valid_start, valid_end = self._cv_split_index(window)
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if valid_end == -1:
            X_test = data_X.iloc[valid_start:]
            Y_test = data_Y.iloc[valid_start:]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, valid_start:valid_end: {valid_start}:{''}")
        else:
            X_test = data_X.iloc[valid_start:(valid_end+1)]
            Y_test = data_Y.iloc[valid_start:(valid_end+1)]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, valid_start:valid_end: {valid_start}:{valid_end+1}")

        return X_train, Y_train, X_test, Y_test
    
    def __recursive_forecast(self, model, history_df, future_df, train_features: List, scaler_features = None, scaler_target = None):
        """
        递归多步预测
        """
        # last 96xday's horizon true targets
        pred_history = list(history_df.iloc[-int(max(self.args.lags)):-1][self.args.target].values)
        # initial features
        current_features_df = history_df[train_features].copy()
        # forecast collection
        predictions = []
        for step in range(self.args.horizon):
            # 初始预测特征
            current_features = current_features_df.iloc[-1].values
            # 预测下一步
            if self.args.scale:
                current_features_scaled = scaler_features.transform(current_features.reshape(1, -1))
                next_pred_scaled = model.predict(current_features_scaled)
                next_pred = scaler_target.inverse_transform(next_pred_scaled.reshape(-1, 1))[0]
            else:
                next_pred = model.predict(current_features.reshape(1, -1))
            predictions.append(next_pred[0])
            # 更新 pred_history
            pred_history.append(next_pred[0])
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1].copy()
            # date, weather features update
            for future_feature in future_df.columns:
                if future_feature != "ds":
                    new_row_df[future_feature] = future_df.iloc[step][future_feature]
            # lag features update
            for i in self.args.lags:
                new_row_df[f"lag_{i}"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, pd.DataFrame([new_row_df])], ignore_index=True)

        return predictions

    def train(self, X_train, Y_train):
        """
        模型训练
        """
        model_params = self.args["model_params"]
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, Y_train)
        
        return model

    def valid(self, model, X_test):
        """
        模型验证
        """
        if len(X_test) > 0:
            Y_pred = model.predict(X_test)
            return Y_pred
        else:
            Y_pred = []
            logger.info(f"X_test length is 0!")
            return Y_pred

    def predict(self, model, X_future):
        """
        模型预测
        """
        if len(X_future) > 0:
            Y_pred = model.predict(X_future)
            return Y_pred
        else:
            Y_pred = []
            logger.info(f"X_test length is 0!")
            return Y_pred

    def recursive_forecast_v2(self, model, initial_features, steps):
        """
        递归多步预测
        """
        preds = []
        current_features = initial_features.copy()
        for _ in range(steps):
            next_pred = self.predict(model, current_features.reshape(1, -1))
            preds.append(next_pred[0])
            current_features = np.roll(current_features, shift=-1)
            current_features[-1] = next_pred
            logger.info(f"predictions: \n{preds}")
        
        return preds

    def recursive_forecast_v1(self, model, X_future):
        preds = np.array([])
        for step in range(self.args["horizon"]):
            logger.info(f'step {step} predict...')
            df_future_x_row = X_future.iloc[step, ].values
            df_future_x_row = np.delete(df_future_x_row, np.where(np.isnan(df_future_x_row)))
            X_future = np.concatenate([df_future_x_row, preds])
            pred_value = self.predict(model, X_future.reshape(1, -1))
            preds = np.concatenate([preds, pred_value])
    
        return preds

    @staticmethod
    def evaluate(Y_test, Y_pred, window: int):
        """
        模型评估
        """
        # 计算模型的性能指标
        if Y_test.mean() == 0:
            Y_test = Y_test.apply(lambda x: x + 0.01)
        res_r2 = r2_score(Y_test, Y_pred)
        res_mse = mean_squared_error(Y_test, Y_pred)
        res_mae = mean_absolute_error(Y_test, Y_pred)
        res_accuracy = 1 - mean_absolute_percentage_error(Y_test, Y_pred)
        # correlation, p_value = pearsonr(Y_test, Y_pred)
        eval_scores = {
            "r2": res_r2,
            "mse": res_mse,
            "mae": res_mae,
            "accuracy": res_accuracy,
            # "correlation": correlation,
        }
        eval_scores = pd.DataFrame(eval_scores, index=[window])
        
        return eval_scores

    def evaluate_result(self, Y_test, Y_pred, window: int):
        """
        测试集预测数据
        """
        start_time = self.args["time_range"]["start_time"]
        now_time = self.args["time_range"]["now_time_start"]
        freq = self.args["freq"]
        # 数据分割指标
        train_start, train_end, valid_start, valid_end = self._cv_split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(start=start_time, end=now_time, freq=freq)})
        if valid_end == -1:
            cv_plot_df_window["ds"] = cv_timestamp_df[valid_start:]#.values
        else:
            cv_plot_df_window["ds"] = cv_timestamp_df[valid_start:(valid_end+1)]#.values
        cv_plot_df_window["train_start"] = [cv_timestamp_df["ds"].values[train_start]] * len(Y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_df["ds"].values[valid_start]] * len(Y_pred)
        cv_plot_df_window["valid_end"] = [cv_timestamp_df["ds"].values[valid_end]] * len(Y_pred)
        cv_plot_df_window["Y_tures"] = Y_test
        cv_plot_df_window["Y_preds"] = Y_pred
        
        return cv_plot_df_window

    def cross_validation(self, data_X, data_Y, n_windows: int, drop_last_window: bool = True):
        """
        交叉验证
        """
        eval_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(n_windows):
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self.cv_split(data_X, data_Y, window)
            logger.info(f"length of X_train: {len(X_train)}, length of Y_train: {len(Y_train)}")
            logger.info(f"length of X_test: {len(X_test)}, length of Y_test: {len(Y_test)}")
            if len(X_train) == 0:
                break
            # 模型训练
            model = self.train(X_train, Y_train)
            # 模型验证
            Y_pred = self.valid(model, X_test)
            # 模型评价
            eval_scores = self.evaluate(Y_test, Y_pred, window)
            eval_scores_df = pd.concat([eval_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self.evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        
        return eval_scores_df, cv_plot_df

    # TODO 早停机制，模型保存
    def model_save(self):
        pass

    def training(self):
        # ------------------------------
        # 模型训练、验证
        # ------------------------------
        if self.args["is_training"]:
            # 数据处理
            logger.info(f"history data process...")
            df_history_X, df_history_Y = self.process_input_history_data()
            
            # 模型训练、评价
            logger.info(f"model training...")
            eval_scores_df, cv_plot_df = self.cross_validation(
                df_history_X,
                df_history_Y,
                n_windows = self.args["n_windows"],
            )
            logger.info(f"cross validation scores: \n{eval_scores_df}")

            # 模型评价指标数据处理
            eval_scores_df = eval_scores_df.mean()
            eval_scores_df = eval_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
            logger.info(f"cross validation average scores: \n{eval_scores_df}")

            # 模型重新训练
            final_model = self.train(X_train=df_history_X, Y_train=df_history_Y)
            logger.info(f"model training over...")
        
        return final_model, df_history_X, cv_plot_df, eval_scores_df
    
    def forecasting_original(self, final_model, df_history_X):
        # ------------------------------
        # 模型预测
        # ------------------------------
        if self.args["is_predicting"]: 
            # 数据处理
            logger.info(f"future data process...")
            df_future_X = self.process_input_future_data()

            # 模型预测
            logger.info(f"model predict...")
            if self.args["pred_method"] == "multip-step-directly":  # 模型多步直接预测(无滞后特征)
                pred_df = self.predict(
                    model = final_model, 
                    X_future = df_future_X,
                )
            elif self.args["pred_method"] == "multip-step-recursion v1":  # 模型多步递归预测
                pred_df = self.recursive_forecast_v1(
                    model=final_model,
                    X_future = df_future_X,
                )
            elif self.args["pred_method"] == "multip-step-recursion v2":  # 模型多步递归预测
                initial_features = df_history_X.iloc[-1].values
                pred_df = self.recursive_forecast_v2(
                    model = final_model, 
                    initial_features = initial_features, 
                    steps = self.args["horizon"]
                )
            logger.info(f"model predict over...")
 
    def run(self):
        # 模型训练
        final_model, df_history_X, cv_plot_df, eval_scores_df = self.training()
        # 模型测试
        pred_df = self.forecast_v1(final_model, df_history_X)
        # ------------------------------
        # 模型输出
        # ------------------------------
        if self.args["is_training"] and not self.args["is_predicting"]:
            return (None, None, cv_plot_df)
        if not self.args["is_training"] and self.args["is_predicting"]:
            return (pred_df, None, None)
        if self.args["is_training"] and self.args["is_predicting"]:
            return (pred_df, eval_scores_df, cv_plot_df)
    
    # TODO
    def run_v1(self):
        # datal load
        input_data = self.__load_data() 
        # model training
        lgb_model, test_scores = self.training_v1(input_data)
        # forecasting
        df_future = self.forecasting_v1(lgb_model, input_data)
        # output data process
        df_future = self.__process_output(df_future)

        return df_future, test_scores  



# 测试代码 main 函数
def main():
    # input data
    input_data = None
    lgb_params = None
    # params
    args = {
        "train_start_time": "2022-01-01 00:00:00",
        "train_end_time": "2022-02-01 00:00:00",
        "forecast_end_time": "2022-02-01 00:00:00",
        "target": "y",
        "freq": "15min",
        "lags": [1, 2, 3, 4],
        "scale": True,
        "horizon": 96,
        "split_length": 96,
        "lgb_params": lgb_params,
    } 
    # model
    model = Model(args)
    # model training and forecasting
    df_power_future, test_scores, features_corr = model.run(
        input_data,
        lgbm_params = args["lgbm_params"],
    )
    # ------------------------------
    # TODO
    # ------------------------------
    """
    logger.info("=" * 50)
    logger.info("Load parameters for traning...")
    logger.info("=" * 50)
    # project params
    project = "ashichuang"
    node = "asc1"
    # data params
    now = datetime.datetime(2025, 2, 6, 23, 46, 0)  # 模型预测的日期时间
    history_days = 30  # 历史数据天数
    predict_days = 1  # 预测未来1天的功率
    freq = "15min"  # 数据频率
    target = self.args.target  # 预测目标变量名称
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
    args = {
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
    input_data = __load_data(data_dir=data_path, data_cfgs=args)
    # ------------------------------
    # data preprocessing
    # ------------------------------
    logger.info("=" * 50)
    logger.info(f"Processing history and future data for training...")
    logger.info("=" * 50)
    logger.info(f"Processing history data for training...")
    logger.info("-" * 40)
    data_history, data_date_history = process_history_data(input_data=input_data, data_cfgs=args)
    logger.info(f"Processing future data for training...")
    logger.info("-" * 40)
    data_future, data_date_future = process_future_data(input_data=input_data, data_cfgs=args)
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
    training_predicting(args, df_history, df_future)

    # ------------------------------
    # logger.info(f"Model training and multip-step-recursion predict...")
    # training_predicting(args, df_history, df_future)
    """
    
if __name__ == "__main__":
    main()
