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
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.feature_engine_todo import (
    extend_datetime_feature,
    extend_datetype_feature,
    extend_lag_feature_univariate,
    extend_lag_feature_multivariate,
    extend_weather_feature,
    extend_future_weather_feature,
)
from utils.model_save_load import ModelDeployPkl
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    def __init__(self, 
                 args: Dict, 
                 history_series = None, weather_history_data = None, date_history_data = None, 
                 future_series = None, weather_future_data = None, date_future_data = None) -> None:
        self.args = args
        # datetime index
        self.train_start_time_str = self.args.train_start_time.strftime("%Y%m%d")
        self.train_end_time_str = self.args.train_end_time.strftime("%Y%m%d")
        self.forecast_start_time_str = self.args.forecast_start_time.strftime("%Y%m%d")
        self.forecast_end_time_str = self.args.forecast_end_time.strftime("%Y%m%d")
        # data
        self.history_series = history_series
        self.weather_history_data = weather_history_data
        self.date_history_data = date_history_data
        self.future_series = future_series
        self.weather_future_data = weather_future_data
        self.date_future_data = date_future_data
        self.freq = self.args.freq

    def _get_history_data(self, timestamp_feat: str):
        """
        历史数据处理
        """ 
        # 构造时间戳完整的历史数据：生成以 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_history = pd.DataFrame({
            "ds": pd.date_range(
                self.train_start_time_str, self.train_end_time_str, 
                freq=self.freq, inclusive="left"
            )
        })
        # 复制 history data
        df = copy.deepcopy(self.history_data)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df[timestamp_feat] = pd.to_datetime(df[timestamp_feat])
            # 去除重复时间戳
            df.drop_duplicates(subset=timestamp_feat, keep="last", inplace=True, ignore_index=True) 
            # 数据处理
            for col in df.columns:
                if col != timestamp_feat:
                    # 将数据转换为字符串类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, y, exogenous_features]
                    df_history[col] = df_history["ds"].map(df.set_index(timestamp_feat)[col])
        
        return df_history

    def _get_future_data(self, timestamp_feat: str):
        """
        未来数据处理
        """
        # 构造时间戳完整的历史数据：生成未来 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_future = pd.DataFrame({
            "ds": pd.date_range(
                self.forecast_start_time_str, self.forecast_end_time_str, 
                freq=self.freq, inclusive="left"
            )
        })
        # 复制 future data
        df = copy.deepcopy(self.future_data)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df[timestamp_feat] = pd.to_datetime(df[timestamp_feat])
            # 去除重复时间戳
            df.drop_duplicates(subset=timestamp_feat, keep="last", inplace=True, ignore_index=True) 
            # 数据处理
            for col in df.columns:
                if col != timestamp_feat:
                    # 将数据转换为字符串类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, exogenous_features]
                    df_future[col] = df_future["ds"].map(df.set_index(timestamp_feat)[col])
        
        return df_future

    def process_input_history_data(self, returnXy: bool = True):
        """
        处理输入历史数据
        """
        # 数据预处理
        df_history = self._get_history_data(timestamp_feat=self.args.timestamp_feat)
        # 特征工程: 外生特征
        exogenous_features = [col for col in df_history.columns if col != "ds" and col != self.args.target]
        # 特征工程: 日期时间特征 
        df_history, datetime_features = extend_datetime_feature(
            df_history, 
            feature_names = [
                'minute', 
                'hour', 'day', 'weekday', 'week', 
                'day_of_week', 'week_of_year', 'month', 'days_in_month', 
                'quarter', 'day_of_year', 'year'
            ],
        )
        # 特征工程: 滞后特征
        if exogenous_features == []:
            df_history, lag_features = extend_lag_feature_univariate(
                df_history, 
                target=self.args.target, 
                lags=self.args.lags,
            )
        else:
            df_history, lag_features, target_features = extend_lag_feature_multivariate(
                df_history, 
                target=self.args.target, 
                exogenous_features=exogenous_features,
                n_lags=self.args.n_lags,
            )
        # 缺失值处理
        df_history = df_history.interpolate()  # 缺失值插值填充
        df_history.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        # 特征排序
        predict_features = lag_features + datetime_features
        if exogenous_features == []:
            df_history = df_history[["ds"] + predict_features + [self.args.target]]
        else:
            df_history = df_history[["ds"] + predict_features + target_features]
        # return all data
        if not returnXy:
            return df_history
        # 预测特征、目标变量分割
        df_history_X = df_history[predict_features]

        if exogenous_features == []:
            df_history_Y = df_history[self.args.target]
        else:
            df_history_Y = df_history[target_features]
        
        return df_history_X, df_history_Y

    def process_input_future_data(self, returnXy: bool = False):
        """
        处理输入未来数据
        """
        # 数据预处理
        df_future = self._get_future_data(timestamp_feat=self.args.timestamp_feat)
        # 特征工程: 外生特征
        exogenous_features = [col for col in df_future.columns if col != "ds" and col != self.args.target]
        # logger.info(f"debug::df_future: \n{df_future}  \ndf_future.columns: {df_future.columns} \nexogenous_features: {exogenous_features}")
        # 特征工程: 日期时间特征
        df_future, datetime_features = extend_datetime_feature(
            df_future,
            feature_names = [
                'minute', 
                'hour', 'day', 
                'weekday', 'week', 'day_of_week', 'week_of_year', 
                'month', 'days_in_month', 'quarter', 
                'day_of_year', 'year'
            ]
        )
        # logger.info(f"debug::df_future: \n{df_future}  \ndf_future.columns: {df_future.columns} \ndatetime_features: {datetime_features}")
        # 特征工程: 滞后特征
        lag_features = []
        # logger.info(f"debug::df_future: \n{df_future}  \ndf_future.columns: {df_future.columns} \nlag_features: {lag_features}")
         # 缺失值处理
        df_future = df_future.interpolate()  # 缺失值插值填充
        df_future.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        # logger.info(f"debug::df_future: \n{df_future}")
        # 预测特征数据
        predict_features = lag_features + datetime_features
        df_future_X = df_future[predict_features]

        return df_future_X, predict_features

    def train(self, X_train, Y_train, X_test, Y_test):
        """
        模型训练
        """
        # 特征列表
        feature_list = X_train.columns
        # 训练集、测试集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        Y_test_df = Y_test.copy()
        # 归一化/标准化
        if self.args.scale:
            scaler_features_test = StandardScaler()
            X_train[feature_list] = scaler_features_test.fit_transform(X_train[feature_list])
            X_test[feature_list] = scaler_features_test.transform(X_test[feature_list])
        else:
            scaler_features_test = None
        # logger.info(f"X_train: \n{X_train} \nX_train.shape: {X_train.shape}")
        # logger.info(f"X_test: \n{X_test} \nX_test.shape: {X_test.shape}")
        # logger.info(f"Y_train: \n{Y_train} \nY_train.shape: {Y_train.shape}")
        # 模型训练
        if isinstance(Y_train, pd.Series):
            Y_train = Y_train.to_frame()
        
        if Y_train.shape[1] == 1:
            model = lgb.LGBMRegressor(**self.args.model_params)
            model.fit(X_train, Y_train)
        else:
            model = MultiOutputRegressor(lgb.LGBMRegressor(**self.args.model_params))
            model.fit(X_train, Y_train)
        
        # 模型预测
        if self.args.pred_method == "univariate-multistep-directly":
            Y_predicted = self.univariate_directly_forecast(
                model = model,
                X_test = X_test,
            )
        elif self.args.pred_method == "univariate-multistep-recursive":
            Y_predicted = self.univariate_recursive_forecast(
                model = model,
                history = pd.concat([X_train_df, Y_train_df], axis=1),
                future = X_test_df,
                lags = self.args.lags,
                steps = self.args.horizon,
                scaler_features = scaler_features_test,
            )
        elif self.args.pred_method == "multivariate-multistep-recursive":
            Y_predicted = self.multivariate_recursive_forecast(
                model = model,
                X_train = X_train_df, 
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.args.lags,
                steps = self.args.horizon,
                scaler_features = scaler_features_test,
            )
        
        return Y_predicted

    def univariate_directly_forecast(self, model, X_test):
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

    def univariate_recursive_forecast(self, model, history, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        # last 96xday's steps true targets
        pred_history = list(history.iloc[-int(max(lags)):-1][self.args.target].values)
        # initial features
        training_feature_list = [col for col in history.columns if col not in ["ds", self.args.target]]
        current_features_df = history[training_feature_list].copy()
        # forecast collection
        predictions = []
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
            pred_history.append(next_pred[0])
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            for i in lags:
                if i > len(pred_history): break
                new_row_df[f"lag_{i}"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # 收集预测结果
            predictions.append(next_pred[0])

        return predictions

    def multivariate_recursive_forecast(self, model, X_train, Y_train, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        # last 96xday's steps true targets
        pred_history = list(Y_train.iloc[-int(max(lags)):].values)
        # initial features
        training_feature_list = [col for col in X_train.columns if col != "ds"]
        current_features_df = X_train[training_feature_list].copy()
        # forecast collection
        predictions = []
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
            pred_history_list = np.array(pred_history[-self.args.n_lags:]).T.flatten().tolist()
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            new_row_df.iloc[:, 0:(Y_train.shape[1]*self.args.n_lags)] = pred_history_list
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # 收集预测结果
            predictions.append(next_pred[0][-1])

        return predictions

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
        X_train = data_X.loc[train_start:train_end]
        Y_train = data_Y.loc[train_start:train_end]
        X_test = data_X.loc[test_start:test_end]
        Y_test = data_Y.loc[test_start:test_end]
        logger.info(f"debug::split indexes: train_start:train_end: {train_start}:{train_end}")
        logger.info(f"debug::split indexes: test_start:test_end: {test_start}:{test_end}")

        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def _evaluate_score(Y_test, Y_pred, window: int):
        """
        模型评估
        """
        # 计算模型的性能指标
        if Y_test.mean() == 0:
            Y_test = Y_test.apply(lambda x: x + 0.01)
        eval_scores = {
            "R2": r2_score(Y_test, Y_pred),
            "mse": mean_squared_error(Y_test, Y_pred),
            "rmse": root_mean_squared_error(Y_test, Y_pred),
            "mae": mean_absolute_error(Y_test, Y_pred),
            "mape": mean_absolute_percentage_error(Y_test, Y_pred),
            "mape accuracy": 1 - mean_absolute_percentage_error(Y_test, Y_pred) 
        }
        eval_scores = pd.DataFrame(eval_scores, index=[window])
        
        return eval_scores

    def _evaluate_result(self, Y_test, Y_pred, window: int):
        """
        测试集预测数据收集
        """ 
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 训练数据完整时间戳
        cv_timestamp_df = pd.DataFrame({
            "ds": pd.date_range(self.train_start_time_str, self.forecast_end_time_str, freq=self.freq, inclusive="left")
        })
        cv_timestamp_ds = cv_timestamp_df["ds"].values
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_plot_df_window["ds"] = cv_timestamp_ds[test_start:test_end]
        cv_plot_df_window["train_start"] = [cv_timestamp_ds[train_start]] * len(Y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_ds[test_start]] * len(Y_pred)
        cv_plot_df_window["test_end"] = [cv_timestamp_ds[test_end]] * len(Y_pred)
        cv_plot_df_window["Y_trues"] = Y_test
        cv_plot_df_window["Y_preds"] = Y_pred
        
        return cv_plot_df_window

    def sliding_window_test(self, data_X, data_Y, n_windows: int, drop_last_window: bool = True):
        """
        交叉验证
        """
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, n_windows + 1):
            logger.info(f"{'-' * 40}")
            logger.info(f"training window: {window}...")
            logger.info(f"{'-' * 40}")
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self._evaluate_split(data_X, data_Y, window)
            logger.info(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
            # 模型训练
            Y_pred = self.train(X_train, Y_train, X_test, Y_test)
            # 模型评价
            if isinstance(Y_test, pd.Series):
                Y_test = Y_test.to_frame()
            Y_test = Y_test.iloc[:, -1:].values
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

    def final_train(self, X_train, Y_train):
        """
        最终模型
        """
        # 特征列表
        feature_list = X_train.columns
        # 所有训练数据
        final_X_train = X_train.copy()
        final_Y_train = Y_train.copy()
        # 归一化/标准化
        if self.args.scale:
            scaler_features = StandardScaler()
            final_X_train[feature_list] = scaler_features.fit_transform(final_X_train[feature_list])
        else:
            scaler_features = None
        # 模型训练
        if isinstance(Y_train, pd.Series):
            Y_train = Y_train.to_frame()
        
        if Y_train.shape[1] == 1:
            final_model = lgb.LGBMRegressor(**self.args.model_params)
            final_model.fit(final_X_train, final_Y_train)
        else:
            final_model = RegressorChain(lgb.LGBMRegressor(**self.args.model_params))
            final_model.fit(final_X_train, final_Y_train)

        return final_model, scaler_features

    def forecast(self, model, df_train_X, df_train_Y, scaler_features):
        # 未来数据处理
        logger.info(f"future data process...")
        df_future_X, future_feature_list = self.process_input_future_data()
        logger.info(f"df_future_X: \n{df_future_X} \ndf_future_X.columns: \n{df_future_X.columns}")
        logger.info(f"future_feature_list: \n{future_feature_list}")
        # 预测特征
        df_future_X = df_future_X.iloc[-self.args.horizon:, ]
        X_future = df_future_X.loc[:, future_feature_list]
        logger.info(f"X_future.head(): \n {X_future.head()} \nX_future length: {len(X_future)} \nX_future.columns: {X_future.columns}")
        # 模型预测
        if isinstance(df_train_Y, pd.Series):
            df_train_Y = df_train_Y.to_frame()
        # directly multi-step forecast
        if self.args.pred_method == "univariate-multistep-directly":
            pred_df = self.univariate_directly_forecast(model, X_future)
        # recursive multi-step forecast
        elif self.args.pred_method == "univariate-multistep-recursive":
            pred_df = self.univariate_recursive_forecast(
                model=model,
                history=pd.concat([df_train_X, df_train_Y], axis=1),
                future=X_future,
                lags=self.args.lags,
                steps=min(self.args.horizon, len(X_future)),
                scaler_features=scaler_features,
            )
        elif self.args.pred_method == "multivariate-multistep-recursive":
            pred_df = self.multivariate_recursive_forecast(
                model=model,
                X_train=df_train_X, 
                Y_train=df_train_Y,
                future=X_future,
                lags=self.args.lags,
                steps=self.args.horizon,
                scaler_features=scaler_features,
            )
        
        return pred_df

    def run(self): 
        # ------------------------------
        # 模型训练、测试
        # ------------------------------
        # 历史数据处理
        logger.info(f"{80*'='}")
        logger.info(f"history data process...")
        logger.info(f"{80*'='}")
        df_history_X, df_history_Y = self.process_input_history_data()
        logger.info(f"df_history_X: \n{df_history_X} \ndf_history_X.columns: \n{df_history_X.columns}")
        logger.info(f"df_history_Y: \n{df_history_Y}")
        
        # 模型训练、评价
        logger.info(f"{80*'='}")
        logger.info(f"model training and testing...")
        logger.info(f"{80*'='}")
        test_scores_df, cv_plot_df = self.sliding_window_test(
            df_history_X,
            df_history_Y,
            n_windows = self.args.n_windows,
        )
        logger.info(f"model training and testing over...")

        # ------------------------------
        # 模型推理
        # ------------------------------
        # 模型重新训练
        logger.info(f"{80*'='}")
        logger.info(f"model finally training...")
        logger.info(f"{80*'='}")
        final_model, scaler_features = self.final_train(
            df_history_X, 
            df_history_Y
        )
        logger.info(f"model finally training over...")
        
        # 模型预测
        logger.info(f"{80*'='}")
        logger.info(f"model forecasting...")
        logger.info(f"{80*'='}")
        pred_df = self.forecast(final_model, df_history_X, df_history_Y, scaler_features)
        logger.info(f"model forecasting result: \n{pred_df}")
        logger.info(f"model forecasting over...")

        return test_scores_df, cv_plot_df, pred_df




# 测试代码 main 函数
def main():
    import datetime
    # ------------------------------
    # params
    # ------------------------------
    # input info
    pred_method = "univariate-multistep-directly"                                  # 预测方法
    freq = "5min"                                                                  # 数据频率
    n_per_day = 24 * 12                                                            # 每天样本数量
    target = "laod"                                                                # 预测目标变量名称
    timestamp_feat = "date"                                                        # 历史数据时间戳特征名称
    # 滞后特征构造
    if pred_method == "univariate-multistep-directly":
        lags = []
    elif pred_method == "univariate-multistep-recursive":
        lags = [3, 2, 1]
    elif pred_method == "multivariate-multistep-recursive":
        lags = list(range(1, 11))
    n_lags = len(lags)
    history_days = 30                                                              # 历史数据天数
    predict_days = 1                                                               # 预测未来1天的功率
    window_len = 8 * n_per_day                                                     # 测试窗口数据长度(训练+测试)
    n_windows = history_days - (window_len - 1)                                    # 测试滑动窗口数量, >=1, 1: 单个窗口
    horizon = predict_days * n_per_day                                             # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    scale = True                                                                   # 是否进行标准化
    now = datetime.datetime(2025, 5, 19, 0, 0, 0)                                  # 模型预测的日期时间
    now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
    start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
    future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻
    logger.info(f"\nhistory data: {start_time} ~ {now_time} \npredict data: {now_time} ~ {future_time}")
    # 模型参数
    model_cfgs = {
        "pred_method": pred_method,
        "freq": freq,
        "target": target,
        "timestamp_feat": timestamp_feat,
        "lags": lags,
        "n_lags": n_lags,
        "n_windows": n_windows,
        "window_len": window_len,
        "horizon": horizon,
        "scale": scale,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "model_params": {
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
        },
    }

if __name__ == "__main__":
    main()
