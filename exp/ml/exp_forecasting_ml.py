# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LightGBM_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description :  1. 单变量多步直接预测(数据标准化)
# *                2. 单变量多步递归预测(滞后特征，数据标准化)
# *                3. 多变量多步递归预测(滞后特征，数据标准化)
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)增加 log;
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy
import random
import datetime
from typing import Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# model
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
# model evaluation
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
from utils.ts.feature_engine import (
    extend_datetime_feature,
    extend_datetype_feature,
    extend_weather_feature, 
    extend_future_weather_feature,
    extend_lag_feature_univariate,
    extend_lag_feature_multivariate,
    extend_lag_feature
)
from utils.model_save_load import ModelDeployPkl
# utils
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class Model:

    def __init__(self, args) -> None:
        self.args = args
        # datetime index
        self.args.train_start_time = args.time_range["start_time"]
        self.args.train_end_time = args.time_range["now_time"]
        self.args.forecast_start_time = args.time_range["now_time"]
        self.args.forecast_end_time = args.time_range["future_time"]
        # self.train_start_time_str = self.args.train_start_time.strftime("%Y%m%d")
        # self.train_end_time_str = self.args.train_end_time.strftime("%Y%m%d")
        # self.forecast_start_time_str = self.args.forecast_start_time.strftime("%Y%m%d")
        # self.forecast_end_time_str = self.args.forecast_end_time.strftime("%Y%m%d")
        logger.info(f"train_start_time_str: {self.train_start_time_str}")
        logger.info(f"train_end_time_str: {self.train_end_time_str}")
        logger.info(f"forecast_start_time_str: {self.forecast_start_time_str}")
        logger.info(f"forecast_end_time_str: {self.forecast_end_time_str}")

    def load_csv_data(self):
        """
        数据加载
        """
        # 历史数据
        df_series_history = pd.read_csv(self.args.data_dir.joinpath(f"df_target.csv"), encoding="utf-8")
        df_date_history = pd.read_csv(self.args.data_dir.joinpath(f"df_date_history.csv"), encoding="utf-8")
        df_weather_history = pd.read_csv(self.args.data_dir.joinpath(f"df_weather_history.csv"), encoding="utf-8")
        # 未来数据
        df_series_future = None
        df_date_future = pd.read_csv(self.args.data_dir.joinpath(f"df_date_future.csv"), encoding="utf-8")
        df_weather_future = pd.read_csv(self.args.data_dir.joinpath(f"df_weather_future.csv"), encoding="utf-8")
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

    def __process_df_timestamp(self, df, ts_col: str):
        """
        时序数据时间特征预处理

        Args:
            df (pd.DataFrame): 时间序列数据
            ts_col (str): 原时间戳列
            new_ts_col (str): 新的时间戳列
            new_ts_col (str, optional): 新的时间戳列. Defaults to "time".
        """
        if df is not None:
            # 数据拷贝
            df_processed = copy.deepcopy(df)
            # 转换时间戳类型
            df_processed[ts_col] = pd.to_datetime(df_processed[ts_col])
            # del df_processed[ts_col]
            # 去除重复时间戳
            df_processed.drop_duplicates(subset=ts_col, keep="last", inplace=True, ignore_index=True)
            return df_processed
        else:
            return df

    def __process_target_series(self, df_template, df_series, 
                                col_ts: str, col_numeric: List, col_categorical: List, col_drop: List):
        """
        目标特征数据预处理
        df_template: ["time", "y"]
        """
        if df_series is not None:
            # 目标特征数据转换为浮点数
            if self.args.target in df_series.columns:
                df_series[self.args.target] = df_series[self.args.target].apply(lambda x: float(x))
                # 将原始数据映射到时间戳完整的 df 中
                df_template["y"] = df_template["time"].map(df_series.set_index(col_ts)[self.args.target])
            # 数值特征处理
            for col in col_numeric:
                if col not in [col_ts, self.args.target] + col_drop:
                    # 将数据转换为浮点数类型
                    df_series[col] = df_series[col].apply(lambda x: float(x))
                    # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_numeric]
                    df_template[col] = df_template["time"].map(df_series.set_index(col_ts)[col]) 
            # 类别特征处理
            for col in col_categorical:
                if col not in [col_ts, self.args.target] + col_drop:
                    # TODO 类别特征处理
                    df_series[col] = self.__categorical_feature_engineering(df_series, col)
                    # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_categorical]
                    df_template[col] = df_template["time"].map(df_series.set_index(col_ts)[col])
        
        return df_template

    # TODO
    def __categorical_feature_engineering(self, df, col):
        pass
    
    def process_history_data(self, input_data: Dict = None):
        """
        历史数据预处理
        """
        # 历史数据格式
        df_history = pd.DataFrame({
            "time": pd.date_range(self.args.train_start_time, self.args.train_end_time, freq=self.args.freq, inclusive="left"),
            # "unique_id": None,
            # "y": None,
        })
        # 特征工程：目标时间序列特征
        df_series_history = self.__process_df_timestamp(df = input_data["df_series_history"], ts_col = self.args.target_ts_feat)
        df_history = self.__process_target_series(
            df_template = df_history,
            df_series = df_series_history, 
            col_ts = self.args.target_ts_feat,
            col_numeric = self.args.target_series_numeric_features, 
            col_categorical = self.args.target_series_categorical_features,
            col_drop=[],
        )
        # 特征工程：日期时间特征
        df_history, datetime_features = extend_datetime_feature(
            df = df_history, 
            feature_names = [
                'minute', 'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )
        
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_history = self.__process_df_timestamp(df = input_data["df_date_history"], ts_col = self.args.date_ts_feat)
        df_history, date_features = extend_datetype_feature(df = df_history, df_date = df_date_history)
        
        # 特征工程：天气特征
        df_weather_history = self.__process_df_timestamp(df = input_data["df_weather_history"], ts_col = self.args.weather_ts_feat)
        df_history, weather_features = extend_weather_feature(df_history = df_history, df_weather = df_weather_history)
        
        # 特征工程：滞后特征
        df_history, lag_features = extend_lag_feature(
            df = df_history, 
            target = self.args.target, 
            lags = self.args.lags,
        )

        # 插值填充预测缺失值
        df_history = df_history.interpolate()
        # df_history = df_history.ffill()
        # df_history = df_history.bfill()
        df_history.dropna(inplace = True, ignore_index = True)

        # 特征排序
        train_features = lag_features + datetime_features + date_features + weather_features
        df_history = df_history[["ds"] + train_features + ["y"]]

        # 本筛选: 异常值处理
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
        # df_history_X, df_history_Y = df_history[train_features], df_history[self.args.target]

        return df_history, train_features

    def process_future_data(self, input_data):
        """
        处理未来数据
        """
        # 未来数据格式
        df_future = pd.DataFrame({
            "time": pd.date_range(self.args.forecast_start_time, self.args.forecast_end_time, freq=self.args.freq, inclusive="left"),
            # "unique_id": None,
            # "y": None,
        })
        
        # 特征工程：除目标特征外的其他特征
        df_series_future = self.__process_df_timestamp(df = input_data["df_series_future"], ts_col = self.args.target_ts_feat)
        df_future = self.__process_target_series(
            df_template = df_future, 
            df_series = df_series_future, 
            col_ts=self.args.target_ts_feat,
            col_numeric = self.args.target_series_numeric_features, 
            col_categorical = self.args.target_series_categorical_features,
            col_drop=[],
        )

        # 特征工程: 日期时间特征
        df_future, datetime_features = extend_datetime_feature(
            df = df_future, 
            feature_names = [
                'minute', 'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )

        # 特征工程: 日期类型(节假日、特殊事件)特征
        df_date_future = self.__process_df_timestamp(df = input_data["df_date_future"], ts_col = self.args.date_ts_feat)
        df_future, date_features = extend_datetype_feature(df = df_future, df_date = df_date_future)

        # 特征工程: 天气特征
        df_weather_future = self.__process_df_timestamp(df = input_data["df_weather_future"], ts_col = self.args.weather_ts_feat)
        df_future, weather_features = extend_future_weather_feature(df_future = df_future, df_weather_future = df_weather_future)
        
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        # df_history = df_history.ffill()
        # df_history = df_history.bfill()
        df_future.dropna(inplace = True, ignore_index = True) 

        # 特征排序
        future_features = datetime_features + date_features + weather_features
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
        # df_future_X = df_future[future_features]

        return df_future, future_features

    @staticmethod
    def univariate_directly_forecast(model, X_test):
        """
        模型预测
        """
        if len(X_test) > 0:
            Y_pred = model.predict(X_test)
            return Y_pred
        else:
            Y_pred = []
            logger.info(f"X_future length is 0!")
            return Y_pred

    def univariate_recursive_forecast(self, model, X_train, Y_train, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        # last 96xday's steps true targets
        pred_history = list(Y_train.iloc[-int(max(lags)):-1][self.target].values)
        # initial features
        training_feature_list = [col for col in X_train.columns if col != "ds"]
        current_features_df = X_train[training_feature_list].copy()
        # forecast collection
        Y_pred = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            if not isinstance(next_pred[0], np.ndarray):
                pred_history.append(next_pred[0])
            else:
                pred_history.append(next_pred[0][-1])
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            for i in lags:
                if i > len(pred_history): break
                if not self.target_transform:
                    new_row_df[f"{self.target}_{i}"] = pred_history[-i]
                else:
                    new_row_df[f"{self.target}(t-{i-1})"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # 收集预测结果
            Y_pred.append(next_pred[0])

        return Y_pred

    def multivariate_recursive_forecast(self, model, X_train, Y_train, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        # last 96xday's steps true targets
        pred_history = list(Y_train.iloc[-int(max(lags)):].values)
        # initial features
        training_feature_list = [col for col in X_train.columns if col != "ds"]
        current_features_df = X_train[training_feature_list].copy()
        # forecast collection
        Y_pred = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1:].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            pred_history.append(next_pred[0])
            pred_history_list = np.array(pred_history[-self.n_lags:]).T.flatten().tolist()
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            new_row_df.iloc[:, 0:(Y_train.shape[1]*self.n_lags)] = pred_history_list
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # TODO 收集预测结果
            # Y_pred.append(next_pred[0][-1])
            Y_pred.append(next_pred[0])

        return Y_pred

    def _evaluate_split_index(self, window: int):
        """
        数据分割索引构建
        """
        test_end    = -1         + (-self.args.horizon) * (window - 1)
        test_start  = test_end   + (-self.args.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.args.window_len) + 1

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            y_test = data_Y.iloc[test_start:]
            logger.info(f"split indexes:: \ntrain_start:train_end: {train_start}:{train_end} \ntest_start:test_end: {test_start}:{''}")
        else:
            X_test = data_X.iloc[test_start:(test_end+1)]
            y_test = data_Y.iloc[test_start:(test_end+1)]
            logger.info(f"split indexes:: \ntrain_start:train_end: {train_start}:{train_end}, \ntest_start:test_end: {test_start}:{test_end+1}")
        logger.info(f"length of X_train: {len(X_train)}, length of y_train: {len(y_train)}")
        logger.info(f"length of X_test: {len(X_test)}, length of y_test: {len(y_test)}")

        return X_train, y_train, X_test, y_test

    @staticmethod
    def _evaluate_score(y_test, y_pred, window: int = 1):
        """
        模型评估
        """
        # 计算模型的性能指标
        if y_test.mean() == 0:
            y_test = y_test.apply(lambda x: x + 0.01)
        test_scores = {
            "R2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "mape accuracy": 1 - mean_absolute_percentage_error(y_test, y_pred),
        }
        test_scores_df = pd.DataFrame(test_scores, index=[window])
        logger.info(f"test_scores_df: \n{test_scores_df}")
        
        return test_scores_df

    def _evaluate_result(self, y_test, y_pred, window: int):
        """
        测试集预测数据
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(self.args.train_start_time, self.args.train_end_time, self.args.freq, inclusive="left")})
        if test_end == -1:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{''}")
        else:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:(test_end+1)]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{test_end+1}")
        cv_plot_df_window["train_start"] = [cv_timestamp_df["ds"].values[train_start]] * len(y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_df["ds"].values[test_start]] * len(y_pred)
        cv_plot_df_window["test_end"] = [cv_timestamp_df["ds"].values[test_end]] * len(y_pred)
        cv_plot_df_window["Y_trues"] = y_test
        cv_plot_df_window["Y_preds"] = y_pred
        
        return cv_plot_df_window
    
    # TODO
    def __calc_features_corr(self, df, train_features):
        """
        分析预测特征与目标特征的相关性
        """
        features_corr = df[train_features + ['load']].corr()
    
        return features_corr
    
    def _window_test(self, X_train, Y_train, X_test):
        """
        模型训练
        """
        # 特征列表
        feature_list = X_train.columns
        # 训练集、测试集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        # 归一化/标准化
        if self.scale:
            scaler_features_test = StandardScaler()
            X_train[feature_list] = scaler_features_test.fit_transform(X_train[feature_list])
            X_test[feature_list] = scaler_features_test.transform(X_test[feature_list])
        else:
            scaler_features_test = None
        # 模型训练
        if Y_train.shape[1] == 1:
            model = lgb.LGBMRegressor(**self.model_params)
            model.fit(X_train, Y_train)
        else:
            model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_params))
            # model = RegressorChain(lgb.LGBMRegressor(**self.model_params))
            model.fit(X_train, Y_train)
        # 模型预测
        if self.pred_method == "multip-step-directly":
            Y_pred = self.univariate_directly_forecast(
                model = model,
                X_test = X_test,
            )
        elif self.pred_method == "univariate-multip-step-recursive":
            Y_pred = self.univariate_recursive_forecast(
                model = model,
                X_train = X_train_df,
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.lags,
                steps = min(self.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        elif self.pred_method == "multivariate-multip-step-recursive":
            Y_pred = self.multivariate_recursive_forecast(
                model = model,
                X_train = X_train_df, 
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.lags,
                steps = min(self.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        
        return Y_pred

    def test(self, data_X: pd.DataFrame, data_Y):
        """
        交叉验证
        """
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, self.n_windows + 1):
            logger.info(f"{'-' * 40}")
            logger.info(f"training window: {window}...")
            logger.info(f"{'-' * 40}")
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self._evaluate_split(data_X, data_Y, window)
            logger.info(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
            if X_train.shape[0] == 0:
                break
            # 目标特征处理
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            if self.target_transform and not self.target_transform_predict and self.pred_method != "multivariate-multip-step-recursive":
                Y_train = Y_train.iloc[:, -1:]
                Y_train.columns = [self.target]
            # 模型测试
            Y_pred = self._window_test(X_train, Y_train, X_test)
            # 多变量, 单变量直接多步预测: 目标特征转换
            if self.target_transform:
                # test
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_test_ups_output = Y_test.iloc[:, 0:1].values
                Y_test            = Y_test.iloc[:, -1:].values
                Y_test = Y_test * Y_test_ups_output - Y_test_ups_output
                if not self.target_transform_predict:
                    # pred
                    Y_pred = np.array(Y_pred)
                    if len(Y_pred.shape) == 1:
                        Y_pred = Y_pred.reshape(-1, 1)
                    else:
                        Y_pred = pd.DataFrame(Y_pred, columns=data_Y.columns)
                        Y_pred = Y_pred.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_test_ups_output - Y_test_ups_output
                elif self.target_transform_predict:
                    # pred
                    Y_preds = pd.DataFrame(Y_pred, columns=data_Y.columns)
                    Y_pred_ups_output = Y_preds.iloc[:, 0:1].values
                    Y_pred            = Y_preds.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_pred_ups_output - Y_pred_ups_output
            elif not self.target_transform and self.pred_method == "multivariate-multip-step-recursive":
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_preds = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_test = Y_test.iloc[:, -1:].values
                Y_pred = Y_preds.iloc[:, -1:].values
            else:
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_test = Y_test.iloc[:, -1:].values
            # 测试集评价指标
            eval_scores = self._evaluate_score(Y_test, Y_pred, window)
            test_scores_df = pd.concat([test_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self._evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        logger.info(f"cross validation test_scores_df: \n{test_scores_df}")
        logger.info(f"cross validation cv_plot_df: \n{cv_plot_df}")
        # 模型评价指标数据处理
        test_scores_df = test_scores_df.mean()
        test_scores_df = test_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
        logger.info(f"cross validation average scores: \n{test_scores_df}")
        
        return test_scores_df, cv_plot_df
    
    def train(self, data_X: pd.DataFrame, data_Y):
        """
        模型训练
        """
        # 特征列表
        feature_list = data_X.columns
        # 所有训练数据
        final_X_train = data_X.copy()
        final_Y_train = data_Y.copy()
        # 归一化/标准化
        if self.scale:
            scaler_features = StandardScaler()
            final_X_train[feature_list] = scaler_features.fit_transform(final_X_train[feature_list])
        else:
            scaler_features = None
        # 模型训练
        if isinstance(final_Y_train, pd.Series):
            final_model = lgb.LGBMRegressor(**self.model_params)
            final_model.fit(final_X_train, final_Y_train)
        else:
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_params))
            # final_model = RegressorChain(lgb.LGBMRegressor(**self.model_params))
            final_model.fit(final_X_train, final_Y_train)

        return final_model, scaler_features

    def forecast(self, model, data_X, data_Y, data_X_future, data_future=None, scaler_features=None):
        """
        模型预测
        """
        # 模型预测
        if len(data_X_future) > 0:
            # directly multi-step forecast
            if self.pred_method == "multip-step-directly":
                if scaler_features is not None:
                    X_test_future = data_X_future.copy()
                    X_test_future = scaler_features.transform(X_test_future)
                else:
                    X_test_future = data_X_future
                Y_pred = self.univariate_directly_forecast(
                    model = model,
                    X_test = X_test_future
                )
            # recursive multi-step forecast
            elif self.pred_method == "univariate-multip-step-recursive":
                Y_pred = self.univariate_recursive_forecast(
                    model = model,
                    X_train = data_X,
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.lags,
                    steps = min(self.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            elif self.pred_method == "multivariate-multip-step-recursive":
                Y_pred = self.multivariate_recursive_forecast(
                    model = model,
                    X_train = data_X, 
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.lags,
                    steps = min(self.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            # 多变量, 单变量直接多步预测: 目标变量转换
            if self.target_transform:
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_future_ups_output = Y_pred_df.iloc[:, 0:1].values
                Y_pred = Y_pred_df.iloc[:, -1:].values
                Y_pred = Y_pred * Y_future_ups_output - Y_future_ups_output
            elif not self.target_transform and self.pred_method == "multivariate-multip-step-recursive":
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_pred = Y_pred_df.iloc[:, -1:].values
            # 预测结果收集
            data_X_future[self.target] = Y_pred
        else:
            data_X_future[self.target] = np.nan
        logger.info(f"data_X_future after forecast: \n{data_X_future.head()} \ndata_X_future length after forecast: {len(data_X_future)}")
        # 输出结果处理
        # data_X_future.dropna(inplace=True, ignore_index=False)
        # logger.info(f"data_X_future after dropna: \n{data_X_future.head()} \ndata_X_future length after dropna: {len(data_X_future)}")
        if self.date_type is not None:
            return data_X_future
        else:
            data_future = pd.merge(data_future, data_X_future, how="outer")
            return data_future

    # TODO
    def model_save(self, final_model):
        """
        模型保存
        """
        model_deploy = ModelDeployPkl(self.args.save_file_path)
        model_deploy.save_model(final_model)

    # TODO
    def process_output(self, df_future, prediction):
        """
        输出结果处理
        """
        # 模型保存路径
        os.makedirs(self.args.result_path, exist_ok=True)
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
        df_future.to_csv(
            os.path.join(self.args.result_path, "prediction.csv"), 
            encoding="utf_8_sig", 
            index=False,
        )

        return df_future 

    def run(self):
        # ------------------------------
        # 数据加载
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Load history and future data...")
        logger.info(f"{80*'='}")
        input_data = self.load_csv_data()
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model history data preprocessing...")
        logger.info(f"{80*'='}")
        if self.args.date_type is not None:
            data_X_workday, data_Y_workday, \
            data_X_offday, data_Y_offday = self.process_history_data(input_data = input_data)
        else:
            data_X, data_Y = self.process_history_data(input_data = input_data)
        logger.info(f"Model history data preprocessing over...")
        # ------------------------------
        # 未来数据处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model future data preprocessing...")
        logger.info(f"{80*'='}")
        if self.args.date_type is not None:
            data_X_future_workday, \
            data_X_future_offday, \
            df_future = self.process_future_data(input_data = input_data)
        else:
            data_X_future, data_future = self.process_future_data(input_data = input_data)
        logger.info(f"Model future data preprocessing over...")
        # ------------------------------
        # TODO 模型选择/模型调参
        # ------------------------------
        final_model_params = None
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_test:
            logger.info(f"{80*'='}")
            logger.info(f"Model testing...")
            logger.info(f"{80*'='}")
            if self.args.date_type is not None:
                test_scores_df, cv_plot_df = self.test(data_X_workday, data_Y_workday)
                test_scores_df, cv_plot_df = self.test(data_X_offday, data_Y_offday)
            else:
                test_scores_df, cv_plot_df = self.test(data_X, data_Y)
            logger.info(f"Model test scores: \n{test_scores_df}")
            logger.info(f"Model test plot_df: \n{cv_plot_df}")
            logger.info(f"Model testing over...")
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.is_train:
            logger.info(f"{80*'='}")
            logger.info(f"Model training...")
            logger.info(f"{80*'='}")
            if self.args.date_type is not None:
                model_workday, scaler_features_workday = self.train(data_X_workday, data_Y_workday)
                model_offday, scaler_features_offday = self.train(data_X_offday, data_Y_offday)
                model = None
            else:
                model_workday = None
                model_offday = None
                model, scaler_features = self.train(data_X, data_Y)
            
            # 模型保存
            if model_workday is not None:
                self.model_save(model_workday)
            if model_offday is not None:
                self.model_save(model_offday)
            if model is not None:
                self.model_save(model)
            logger.info(f"model saved to path: {self.args.model_save_path}")
            logger.info(f"Model training over...")
        # ------------------------------
        # 模型预测
        # ------------------------------
        if self.args.is_forecast: 
            logger.info(f"{80*'='}")
            logger.info(f"Model Forecast...")
            logger.info(f"{80*'='}")
            if self.date_type is not None:
                pred_df_workday = self.forecast(
                    model_workday,
                    data_X_workday, data_Y_workday,
                    data_X_future_workday, None,
                    scaler_features_workday,
                )
                pred_df_offday = self.forecast(
                    model_offday,
                    data_X_offday, data_Y_offday,
                    data_X_future_offday, None,
                    scaler_features_offday
                )
                df_future = pd.merge(df_future, pred_df_workday, how="outer")
                df_future = pd.merge(df_future, pred_df_offday, how="outer")
            else:
                df_future = self.forecast(
                    model, 
                    data_X, data_Y, 
                    data_X_future, data_future, 
                    scaler_features
                )
            # 模型输出
            logger.info(f"Forecast result processing...")
            y_pred = self.process_output(y_pred)
            logger.info(f"Model forecast result: {y_pred}")
            logger.info(f"Model forecasting over...")


@dataclass
class ModelConfig:
    data_dir = Path("./dataset/electricity/exp_ml")
    # target series
    target = "h_total_use"                        # 预测目标变量名称
    target_ts_feat = "count_data_time"            # 功率数据时间戳特征名称
    target_series_numeric_features = []           # 目标时间序列的数值特征
    target_series_categorical_features = []       # 目标时间序列的类别特征
    freq = "5min"                                 # 数据频率
    # date type
    date_ts_feat = "date"                         # 日期数据时间戳特征名称
    # weather
    weather_ts_feat = "ts"                        # 天气数据时间戳特征名称
    # model params
    pred_method = "multip-step-directly"          # 预测方法
    scale = False                                 # 是否进行标准化
    target_transform = False                      # 预测目标是否需要转换
    target_transform_predict = None               # 预测目标的转换特征是否需要预测
    date_type = None                              # 日期类型，用于区分工作日，非工作日
    lags = []                                     # 特征滞后数列表
    n_lags = len(lags)                            # 特征滞后数个数
    n_per_day = 24 * 4                            # 每天样本数量
    history_days = 19                             # 历史数据天数
    predict_days = 1                              # 预测未来1天的功率
    window_days = 7                               # 滑动窗口天数
    horizon = predict_days * n_per_day            # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    n_windows = history_days - (window_days - 1)  # 测试滑动窗口数量, >=1, 1: 单个窗口
    window_len = window_days * n_per_day if n_windows > 1 else history_days * n_per_day   # 测试窗口数据长度(训练+测试)
    now_time = datetime.datetime(2025, 7, 10, 0, 0, 0).replace(tzinfo=None, minute=0, second=0, microsecond=0)  
    time_range = {
        "start_time": now_time.replace(hour=0) - datetime.timedelta(days=history_days),   # 时间序列历史数据开始时刻
        "now_time": now_time,                                                             # 时间序列当前时刻(模型预测的日期时间)
        "future_time": now_time + datetime.timedelta(days=predict_days),                  # 时间序列未来结束时刻
        "before_days": -history_days,
        "after_days": predict_days,
    }
    model_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "max_bin": 31,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "verbose": -1,
    }




# 测试代码 main 函数
def main():
    args = ModelConfig()
    logger.info(f"args:\n{args.time_range}")

    # model instance
    model = Model(args)

if __name__ == "__main__":
    main()
