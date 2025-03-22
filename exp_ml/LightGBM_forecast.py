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
# model
import lightgbm as lgb
# model evaluation
from scipy.stats import pearsonr
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
# data processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# feature engineering
from utils.feature_engine import (
    extend_datetime_stamp_feature,
    extend_lag_feature,
    extend_date_type_features,
    extend_lag_features,
    extend_datetime_features,
    extend_weather_features,
    extend_future_weather_features,
)
# utils
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    # TODO
    def __init__(self, args: Dict, history_data, future_data) -> None:
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
        df_series_history = pd.read_csv(os.path.join(self.args.data_dir, f"df_target.csv"), encoding="utf-8")
        df_date_history = pd.read_csv(os.path.join(self.args.data_dir, f"df_date_history.csv"), encoding="utf-8")
        df_weather_history = pd.read_csv(os.path.join(self.args.data_dir, f"df_weather_history.csv"), encoding="utf-8")
        # 未来数据
        df_series_future = None
        df_date_future = pd.read_csv(os.path.join(self.args.data_dir, f"df_date_future.csv"), encoding="utf-8")
        df_weather_future = pd.read_csv(os.path.join(self.args.data_dir, f"df_weather_future.csv"), encoding="utf-8")
        # 输入数据以字典形式整理 
        input_data = {
            "df_series_history": df_series_history,
            "df_date_history": df_date_history,
            "df_weather_history": df_weather_history,
            "df_series_future": df_series_future,
            "df_date_future": df_date_future,
            "df_weather_future": df_weather_future,
        }
        
        return input_data

    def __process_df_timestamp(self, df, date_col: str, new_date_col: str = "ds"):
        """
        时序数据预处理

        Args:
            data (pd.DataFrame): 时间序列数据
            date_col (str): 原时间戳列
            new_date_col (str): 新的时间戳列
        """
        # 数据拷贝
        df_processed = copy.deepcopy(df)
        # 转换时间戳类型
        df_processed[new_date_col] = pd.to_datetime(df_processed[date_col])
        del df_processed[date_col]
        # 去除重复时间戳
        df_processed.drop_duplicates(
            subset = new_date_col,
            keep = "last",
            inplace = True,
            ignore_index = True,
        )

        return df_processed

    def __process_target_series(self, df_template, df, col_numeric: List, col_categorical: List):
        """
        目标特征数据预处理
        """
        # 目标特征数据转换为浮点数
        if self.args.target in df.columns:
            df[self.args.target] = df[self.args.target].apply(lambda x: float(x))
            # 将原始数据映射到时间戳完整的 df 中
            df_template["y"] = df_template["ds"].map(df.set_index("ds")[self.args.target])
        # 数值特征处理
        for col in col_numeric:
            if col not in ["ds", self.args.target]:
                # 将数据转换为浮点数类型
                df[col] = df[col].apply(lambda x: float(x))
                # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_numeric]
                df_template[col] = df_template["ds"].map(df.set_index("ds")[col]) 
        # 类别特征处理
        for col in col_categorical:
            if col not in ["ds", self.args.target]:
                # TODO 类别特征处理
                df[col] = self.__categorical_feature_engineering(df, col)
                # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_categorical]
                df_template[col] = df_template["ds"].map(df.set_index("ds")[col])    
        
        return df_template

    # TODO
    def __categorical_feature_engineering(self):
        pass

    # TODO
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
        # 历史数据格式
        df_history = pd.DataFrame({
            "ds": pd.date_range(
                self.args.train_start_time, 
                self.args.train_end_time, 
                freq = self.args.freq
            ),
            "unique_id": None,
            "y": None,
        })
        # TODO 特征工程：目标时间序列特征
        df_series_history = self.__process_df_timestamp(
            input_data["df_series_history"], 
            self.args.df_target_ds_col, 
            "ds"
        )  # count_data_time
        df_history = self.__process_target_series(
            df_history, 
            df_series_history, 
            col_numeric=self.args.target_series_numeric_features, 
            col_categorical=self.args.target_series_categorical_features
        )
        # TODO 特征工程：天气特征
        df_weather_history = self.__process_df_timestamp(
            input_data["df_weather_history"], 
            self.args.df_weather_history_ds_col, 
            "ds"
        )  # ts
        df_history, weather_features = extend_weather_features(df_history, df_weather_history)
        # TODO 特征工程：日期时间特征
        # 日期时间特征
        df_history, datetime_features = extend_datetime_features(
            df_history, 
            feature_names = [
                'minute', 
                'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )
        # TODO 特征工程：日期类型(节假日、特殊事件)特征
        df_date_history = self.__process_df_timestamp(
            input_data["df_date_history"], 
            self.args.df_date_history_ds_col, 
            "ds"
        )  # date
        df_history, date_features = extend_date_type_features(df_history, df_date_history)
        # TODO 特征工程：滞后特征
        df_history, lag_features = extend_lag_features(
            df=df_history, 
            target=self.args.target, 
            lags=self.args.lags
        )
        # df_history, lag_features = extend_lag_features(
        #     df=df_history, 
        #     target=self.args.target, 
        #     group_col=None, 
        #     numLags=self.args.lags,
        #     numHorizon=0, 
        #     dropna=True,
        # )
        # df_future_lags = extend_lag_feature(
        #     df=df_history, 
        #     target=self.args.target, 
        #     group_col=None, 
        #     numLags=0, 
        #     numHorizon=self.args.lags, 
        #     dropna=False,
        # )
        # TODO 插值填充预测缺失值
        df_history = df_history.interpolate()
        df_history = df_history.ffill()
        df_history = df_history.bfill()
        df_history.dropna(inplace = True, ignore_index = True)

        # 特征排序
        train_features = lag_features + \
            weather_features + \
            datetime_features + \
            date_features
        df_history = df_history[["ds"] + train_features + ["y"]]

        # TODO 样本筛选: 异常值处理
        df_history = df_history[df_history["y"] > self.args.threshold]

        # TODO 数据分割: 工作日预测特征，目标特征
        # if self.args.is_workday:
        #     df_history = copy.deepcopy(df_history.query("(date_type == 1) or ((date_type == 2) and (datetime_weekday == 5))"))
        #     df_history_workday_path = os.path.join(self.args.data_path, "df_history_workday.csv")
        #     if not os.path.exists(df_history_workday_path):
        #         df_history.to_csv(df_history_workday_path)
        #         logger.info(f"df_history_workday has saved in {df_history_workday_path}")
        # else:
        #     df_history = copy.deepcopy(df_history.query("(date_type > 2) or ((date_type == 2) and (datetime_weekday == 6))"))
        #     df_history_offday_path = os.path.join(self.args.data_path, "df_history_offday.csv")
        #     if not os.path.exists(df_history_offday_path):
        #         df_history.to_csv(df_history_offday_path)
        #         logger.info(f"df_history_offday has saved in {df_history_offday_path}")

        # TODO 预测特征、目标特征分割
        df_history_X = df_history[train_features]
        df_history_Y = df_history[self.args.target]

        return df_history, train_features

    # TODO 处理滞后特征更新
    def __preprocessing_future_data(self, input_data):
        """
        处理未来数据
        """
        # 未来数据格式
        df_future = pd.DataFrame({
            "ds": pd.date_range(
                self.args.forecast_start_time,
                self.args.forecast_end_time,
                freq = self.args.freq
            ),
            "unique_id": None,
            "y": None,
        })
        # TODO 特征工程：除目标特征外的其他特征
        df_series_future = self.__process_df_timestamp(
            input_data["df_series_future"], 
            self.args.df_series_future_ds_col, 
            "ds"
        )  # count_data_time
        df_future = self.__process_target_series(
            df_future, 
            df_series_future, 
            col_numeric=self.args.target_series_numeric_features, 
            col_categorical=self.args.target_series_categorical_features
        )
        # TODO 特征工程: 天气特征
        df_weather_future = self.__process_df_timestamp(
            input_data["df_weather_future"], 
            self.args.df_weather_future_ds_col, 
            "ds"
        )  # ts
        df_future, weather_features = extend_future_weather_features(df_future, df_weather_future)
        # TODO 特征工程: 日期时间特征
        df_history, datetime_features = extend_datetime_features(
            df_history, 
            feature_names = [
                'minute', 
                'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )
        # TODO 特征工程: 日期类型(节假日、特殊事件)特征
        df_date_future = self.__process_df_timestamp(
            input_data["df_date_future"], 
            self.args.df_date_future_ds_col, 
            "ds"
        )  # date
        df_future, date_features = extend_date_type_features(df_future, df_date_future)
        
        # TODO 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_history = df_history.ffill()
        df_history = df_history.bfill()
        df_future.dropna(inplace = True, ignore_index = True) 

        # 特征排序
        future_features = weather_features + datetime_features + date_features
        df_future = df_future[["ds"] + future_features]
        
        # TODO 数据分割: 工作日预测特征, 目标特征
        # df_future = copy.deepcopy(df_future[df_future["date_type"] == 1])
        # if self.args.predict_days == 1:
        #     df_future_date_type = df_future["date_type"].unique()[0]
        #     df_future_datetime_weekday = df_future["datetime_weekday"].unique()[0]
        #     if df_future_date_type in [1, 2] or df_future_datetime_weekday == 5:
        #         self.args.is_workday = True
        #     elif df_future_date_type >= 2 or df_future_datetime_weekday == 6:
        #         self.args.is_workday = False

        # TODO 预测特征分割
        df_future_X = df_future[future_features]

        return df_future, future_features
 
    def __recursive_forecast(self, model, df_history, df_future, scale: str, scaler_features = None, scaler_target = None):
        """
        递归多步预测
        """
        # last 96xday's steps true targets
        pred_history = list(df_history.iloc[-int(max(self.args.lags)):-1][self.args.target].values)
        # initial features
        training_feature_list = [
            col for col in df_history.columns 
            if col not in ["ds", self.args.target]
        ]
        current_features_df = df_history[training_feature_list].copy()
        # forecast collection
        predictions = []
        # 预测下一步
        self.args.horizon = min(self.args.horizon, len(df_future))
        for step in range(self.args.horizon):
            # 初始预测特征
            if scale == "features_target" or scale == "features":
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            if scale == "features_target":
                next_pred = scaler_target.inverse_transform(next_pred.reshape(-1, 1))[0]  # 特征逆转换
            # 更新 pred_history
            pred_history.append(next_pred[0])
            
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1].copy()
            # date, weather features update
            for future_feature in df_future.columns:
                if future_feature != "ds":
                    new_row_df[future_feature] = df_future.iloc[step][future_feature]
            # lag features update
            for i in self.args.lags:
                if i > len(pred_history): break
                new_row_df[f"lag_{i}"] = pred_history[-i]
            
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, pd.DataFrame([new_row_df])], axis = 0, ignore_index=True)

            # 收集预测结果
            predictions.append(next_pred[0])
        
        return predictions
    
    # TODO
    def training(self, df_history: Dict):
        """
        模型训练、测试
        """
        # 预测特征、目标特征分割
        predict_features = [
            col for col in df_history.columns 
            if col not in ["ds", self.args.target]
        ]
        data_X = df_history[predict_features]
        data_Y = df_history[self.args.target]
        
        # TODO 训练集、测试集划分
        data_length = len(data_X)
        X_train = data_X.iloc[-data_length:-self.args.split_length]
        Y_train = data_Y.iloc[-data_length:-self.args.split_length]
        X_test = data_X.iloc[-self.args.split_length:]
        Y_test = data_Y.iloc[-self.args.split_length:]
        # 备份
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        Y_test_df = Y_test.copy()
        """
        # ------------------------------
        # 模型测试
        # ------------------------------
        # 归一化/标准化
        if self.args.scale == "features_target":
            if self.args.scale_method == "std":
                scaler_features_test = StandardScaler()
                scaler_target_test = StandardScaler()
            elif self.args.scale_method == "minmax":
                scaler_features_test = MinMaxScaler()
                scaler_target_test = MinMaxScaler()
            X_train[predict_features] = scaler_features_test.fit_transform(X_train[predict_features])
            Y_train = scaler_target_test.fit_transform(pd.DataFrame(Y_train))
            Y_train = pd.DataFrame(Y_train, columns=["load"])
            X_test[predict_features] = scaler_features_test.transform(X_test[predict_features])
            Y_test = scaler_target_test.transform(pd.DataFrame(Y_test))
            Y_test = pd.DataFrame(Y_test, columns=["load"])
        elif self.args.scale == "features":
            if self.args.scale_method == "std":
                scaler_features_test = StandardScaler()
            elif self.args.scale_method == "minmax":
                scaler_features_test = MinMaxScaler()
            scaler_target_test = None
            X_train[predict_features] = scaler_features_test.fit_transform(X_train[predict_features])
            X_test[predict_features] = scaler_features_test.transform(X_test[predict_features])
        elif self.args.scale is None:
            scaler_features_test = None
            scaler_target_test = None
        # 模型训练
        lgb_model = lgb.LGBMRegressor(**self.args.model_params)
        lgb_model.fit(X_train, Y_train)
        # 特征重要性排序
        lgb.plot_importance(
            lgb_model, 
            importance_type="gain",  # gain, split
            figsize=(7,6), 
            title="LightGBM Feature Importance (Gain)"
        )
        # plt.show(); 
        # 模型测试
        Y_predicted = self.__recursive_forecast(
            model = lgb_model,
            df_history = pd.concat([X_train_df, Y_train_df], axis=1),
            df_future = X_test_df,
            scale = self.args.scale,
            scaler_features = scaler_features_test,
            scaler_target = scaler_target_test,
        )
        # logger.info(f"Y_predicted: {Y_predicted} \nY_predicted length: {len(Y_predicted)}")
        # 模型评估
        test_scores = {
            "R2": r2_score(Y_test_df, Y_predicted),
            "mse": mean_squared_error(Y_test_df, Y_predicted),
            "rmse": root_mean_squared_error(Y_test_df, Y_predicted),
            "mae": mean_absolute_error(Y_test_df, Y_predicted),
            "mape": mean_absolute_percentage_error(Y_test_df, Y_predicted),
            "accuracy": 1 - mean_absolute_percentage_error(Y_test_df, Y_predicted) 
        }
        logger.info(f"R2: {test_scores['R2']:.4f}")
        logger.info(f"mse: {test_scores['mse']:.4f}")
        logger.info(f"rmse: {test_scores['rmse']:.4f}")
        logger.info(f"mape: {test_scores['mape']:.4f}")
        logger.info(f"mape: {test_scores['mape']:.4f}")
        logger.info(f"mape accuracy: {test_scores['accuracy']:.4f}")
        """
        # ------------------------------
        # 最终模型
        # ------------------------------
        # 所有训练数据
        final_X_train = pd.concat([X_train_df, X_test_df], axis = 0)
        final_Y_train = pd.concat([Y_train_df, Y_test_df], axis = 0)
        # 归一化/标准化
        if self.args.scale == "standard":
            scaler_features= StandardScaler()
        elif self.args.scale == "minmax":
            scaler_features= MinMaxScaler()
            final_X_train[predict_features] = scaler_features.fit_transform(final_X_train[predict_features])
        elif self.args.scale is None:
            scaler_features = None
        # 模型训练
        final_model = lgb.LGBMRegressor(**self.args.model_params)
        final_model.fit(final_X_train, final_Y_train)

        return (
            final_model, 
            # test_scores, 
            scaler_features, 
            scaler_target
        )

    # TODO
    def forecasting(self, lgb_model, df_history, df_future, scaler_features, scaler_target):
        """
        时间序列预测

        Args:
            lgb_model (_type_): _description_
            input_data (_type_): _description_
            scaler_features (_type_): _description_
            scaler_target (_type_): _description_

        Returns:
            _type_: _description_
        """ 
        # multi-step recursive forecast
        predictions = self.__recursive_forecast(
            model = lgb_model,
            df_history = df_history,
            df_future = df_future,
            scale = self.args.scale,
            scaler_features = scaler_features,
            scaler_target = scaler_target,
        )
        df_future[self.args.target] = predictions
        # TODO 输出结果处理
        df_future.dropna(inplace=True, ignore_index=True)

        return df_future 
    # ------------------------------
    # 
    # ------------------------------
    def __cv_split_index(self, window: int):
        """
        数据分割索引构建
        """
        valid_end   = -1        + (-self.args["horizon"]) * window
        valid_start = valid_end + (-self.args["horizon"]) + 1
        train_end   = valid_start
        train_start = valid_end + (-self.args["data_length"]) + 1

        return train_start, train_end, valid_start, valid_end

    def __cv_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, valid_start, valid_end = self.__cv_split_index(window)
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

    def train(self, X_train, Y_train):
        """
        模型训练
        """
        model = lgb.LGBMRegressor(**self.args.model_params)
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
            logger.info(f"X_future length is 0!")
            return Y_pred

    @staticmethod
    def __evaluate(Y_test, Y_pred, window: int):
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

    def __evaluate_result(self, Y_test, Y_pred, window: int):
        """
        测试集预测数据
        """
        start_time = self.args["time_range"]["start_time"]
        now_time = self.args["time_range"]["now_time_start"]
        freq = self.args["freq"]
        # 数据分割指标
        train_start, train_end, valid_start, valid_end = self.__cv_split_index(window)
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
            X_train, Y_train, X_test, Y_test = self.__cv_split(data_X, data_Y, window)
            logger.info(f"length of X_train: {len(X_train)}, length of Y_train: {len(Y_train)}")
            logger.info(f"length of X_test: {len(X_test)}, length of Y_test: {len(Y_test)}")
            if len(X_train) == 0:
                break
            # 模型训练
            model = self.train(X_train, Y_train)
            # 模型验证
            Y_pred = self.valid(model, X_test)
            # 模型评价
            eval_scores = self.__evaluate(Y_test, Y_pred, window)
            eval_scores_df = pd.concat([eval_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self.__evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        logger.info(f"cross validation scores: \n{eval_scores_df}")

        # 模型评价指标数据处理
        eval_scores_df = eval_scores_df.mean()
        eval_scores_df = eval_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
        logger.info(f"cross validation average scores: \n{eval_scores_df}")
        
        return eval_scores_df, cv_plot_df

    def process_output(self, df_future):
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
            "timestamp",
            "predict_value",
            "predict_adjustable_amount",
        ]]

        return df_future
    
    # TODO 早停机制，模型保存
    def __model_save(self):
        pass
    
    # TODO 模型保存
    def __result_save(self, result):
        os.makedirs(self.args.result_path, exist_ok=True)
        result.to_csv(
            os.path.join(self.args.result_path, "prediction.csv"), 
            encoding="utf_8_sig", 
            index=False,
        )

    # TODO
    def run(self):
        # ------------------------------
        # 数据加载
        # ------------------------------
        input_data = self.__load_data()

        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"history data process...")
        df_history, train_features = self.__preprocessing_history_data(input_data = input_data)
        df_history_X, df_history_Y = df_history[train_features], df_history[self.args.target]

        # ------------------------------
        # 模型训练、验证
        # ------------------------------
        if self.args.is_validate:
            logger.info(f"model training...")
            eval_scores_df, cv_plot_df = self.cross_validation(
                df_history_X,
                df_history_Y,
                n_windows = self.args.n_windows,
                drop_last_window = True,
            )
            logger.info(f"cross validation average scores: \n{eval_scores_df}")

        # ------------------------------
        # 模型重新训练
        # ------------------------------
        if self.args.is_train:
            final_model, scaler_features, scaler_target = self.train(
                X_train = df_history_X, 
                Y_train = df_history_Y,
            )
            logger.info(f"model training over...")

        # ------------------------------
        # 未来数据处理
        # ------------------------------
        logger.info(f"future data process...")
        df_future, future_features = self.__preprocessing_future_data(input_data = input_data)
        df_future_X = df_future[future_features]

        # ------------------------------
        # 模型预测
        # ------------------------------
        if self.args.is_forecast:
            logger.info(f"model forecasting...")
            if self.args.pred_method == "multip-step-directly":  # 模型单步预测
                pred_df = self.predict(
                    model = final_model, 
                    X_future = df_future_X,
                )
            elif self.args.pred_method == "multip-step-recursion":  # 模型多步递归预测
                df_future = self.forecasting(
                    lgb_model = final_model,
                    df_history = df_history,
                    df_future = df_future,
                    scaler_features = scaler_features,
                    scaler_target = scaler_target,
                )
                # 模型输出
                df_future = self.process_output(df_future)
            elif self.args.pred_method == "multip-step-directly-lags":  # TODO 模型多步直接预测
                pred_df = []
            logger.info(f"model predict result: \n{pred_df}")
            logger.info(f"model predict over...")




# 测试代码 main 函数
def main():
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
        "lgb_params": None,
    }
    """
    # ------------------------------
    # TODO
    # ------------------------------
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
