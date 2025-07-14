# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LightGBM_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)增加 log;
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy
import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.metrics import (
    r2_score,  # R2
    mean_squared_error,  # MSE
    root_mean_squared_error,  # RMSE
    mean_absolute_error,  # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import StandardScaler

from utils.feature_engine import (
    extend_datetime_feature,
    extend_lag_feature_univariate,
    extend_lag_feature_multivariate,
    extend_date_type_feature,
    extend_weather_feature,
    extend_future_weather_feature,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model:

    def __init__(self, model_cfgs: Dict, history_data, df_date, df_weather, future_data, df_date_future, df_weather_future):
        self.model_cfgs = model_cfgs
        self.history_data = history_data
        self.df_date = df_date
        self.df_weather = df_weather
        self.future_data = future_data
        self.df_date_future = df_date_future
        self.df_weather_future = df_weather_future
        self.start_time = self.model_cfgs.time_range["start_time"]
        self.now_time = self.model_cfgs.time_range["now_time"]
        self.future_time = self.model_cfgs.time_range["future_time"]

    def _preprocess_data(self, raw_df: pd.DataFrame, column_name: str, new_column_name: str):
        # 复制 future data
        df = copy.deepcopy(raw_df)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df[new_column_name] = pd.to_datetime(df[column_name])
            # 去除重复时间戳
            df.drop_duplicates(subset=new_column_name, keep="last", inplace=True, ignore_index=True)

        return df

    def _get_history_data(self, timestamp_feat: str):
        """
        历史数据处理
        """
        # 构造时间戳完整的历史数据：生成以 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_history = pd.DataFrame({
            "timeStamp": pd.date_range(self.start_time, self.now_time, freq=self.model_cfgs.freq, inclusive="left")
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
                if col not in [timestamp_feat, "date"]:
                    # 将数据转换为字符串类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, y, exogenous_features]
                    df_history[col] = df_history["timeStamp"].map(df.set_index(timestamp_feat)[col])
        
        return df_history

    def _get_future_data(self, timestamp_feat: str):
        """
        未来数据处理
        """
        # 构造时间戳完整的历史数据：生成未来 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_future = pd.DataFrame({
            "timeStamp": pd.date_range(
                self.now_time.replace(minute=0, second=0, microsecond=0), self.future_time, 
                freq=self.model_cfgs.freq, inclusive="left"
        )})
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
                    df_future[col] = df_future["timeStamp"].map(df.set_index(timestamp_feat)[col])
        
        return df_future

    def _process_data_history(self):
        # 数据预处理
        df_history = self._get_history_data(timestamp_feat=self.model_cfgs.power_ts_feat)
        logger.info(f"df_history: \n{df_history}")
        # 删除含空值的行
        df_history.dropna(inplace=True, ignore_index=True)
        logger.info(f"df_history shape after drop NA: {df_history.shape}")
        # 删除需求负荷小于 0 的样本
        df_history = df_history[df_history[self.model_cfgs.target] > 0]
        logger.info(f"df_history shape after data filter: {df_history.shape}")
        logger.info(f"df_history has nan or not: \n{df_history.isna().any()}")
        # 外生特征
        exogenous_features = [col for col in df_history.columns if col != "timeStamp" and col != self.model_cfgs.target]
        logger.info(f"df_history exogenous_features: {exogenous_features}")
        # 特征工程: 日期时间特征
        df_history, datetime_features = extend_datetime_feature(
            df_history,  feature_names = ['minute', 'hour', 'day', 'weekday', 'week', 'day_of_week', 'week_of_year', 'month', 'days_in_month', 'quarter', 'day_of_year', 'year'],
        )
        logger.info(f"df_history shape after merge datetime features: {df_history.shape} \ndf_history.columns: {df_history.columns}")

        return df_history, exogenous_features, datetime_features

    def _process_date_history(self, df_history):
        df_date = self._preprocess_data(self.df_date, self.model_cfgs.date_ts_feat, "timeStamp")
        logger.info(f"df_date: \n{df_date}")
        # 特征工程: 日期特征
        if self.df_date is not None:
            df_history, date_features = extend_date_type_feature(df_history, self.df_date)
            logger.info(f"df_history shape after merge date features: {df_history.shape} \ndf_history.columns: {df_history.columns}")
        else:
            date_features = []
        
        return df_history, date_features
    
    def _process_weather_history(self, df_history):
        df_weather = self._preprocess_data(self.df_weather, self.model_cfgs.weather_ts_feat, "timeStamp")
        logger.info(f"df_weather: \n{df_weather}")
         # 特征工程: 天气特征
        if self.df_weather is not None:
            df_history, weather_features = extend_weather_feature(df_history, self.df_weather)
            logger.info(f"df_history shape after merge weather features: {df_history.shape} \ndf_history.columns: {df_history.columns}")
        else:
            weather_features = []
        
        return df_history, weather_features

    def _process_data_future(self):
        # 数据预处理
        df_future = self._get_future_data(timestamp_feat=self.model_cfgs.power_ts_feat)
        logger.info(f"df_future: \n{df_future}")
        # 特征工程: 外生特征
        exogenous_features = [col for col in df_future.columns if col != "timeStamp" and col != self.model_cfgs.target]
        logger.info(f"df_load exogenous_features: {exogenous_features}")
        # 特征工程: 日期时间特征
        df_future, datetime_features = extend_datetime_feature(
            df_future, 
            feature_names = ['minute', 'hour', 'day', 'weekday', 'week', 'day_of_week', 'week_of_year', 'month', 'days_in_month', 'quarter', 'day_of_year', 'year']
        )
        logger.info(f"df_future shape after merge datetime features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
        
        return df_future, exogenous_features, datetime_features

    def _process_date_future(self, df_future):
        df_date_future = self._preprocess_data(self.df_date_future, self.model_cfgs.date_ts_feat, "timeStamp")
        logger.info(f"df_date_future: \n{df_date_future}")
        # 特征工程: 日期特征
        if df_date_future is not None:
            df_future, date_features = extend_date_type_feature(df_future, df_date_future)
        else:
            date_features = []
        logger.info(f"df_future shape after merge date features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
    
        return df_future, date_features
    
    def _process_weather_future(self, df_future):
        df_weather_future = self._preprocess_data(self.df_weather_future, self.model_cfgs.weather_ts_feat, "timeStamp")
        logger.info(f"df_weather_future shape: \n{df_weather_future}")
        # 特征工程: 环境特征
        if df_weather_future is not None:
            df_future, weather_features = extend_future_weather_feature(df_future, df_weather_future)
        else:
            weather_features = []
        logger.info(f"df_future shape after merge weather features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
        
        return df_future, weather_features

    def process_history_data(self):
        """
        处理输入历史数据
        """
        # 历史数据
        df_history, exogenous_features, datetime_features = self._process_data_history()
        df_history, date_features = self._process_date_history(df_history)
        df_history, weather_features = self._process_weather_history(df_history)
        
        # 特征工程: 滞后特征
        if exogenous_features == [] or (not self.model_cfgs.target_transform and self.model_cfgs.pred_method != "multivariate-multip-step-recursive"):
            df_history, lag_features = extend_lag_feature_univariate(
                df=df_history,
                target=self.model_cfgs.target, 
                lags=self.model_cfgs.lags,
            )
        else:
            df_history, lag_features, target_features = extend_lag_feature_multivariate(
                df=df_history,
                exogenous_features=exogenous_features,
                target=self.model_cfgs.target,
                n_lags=self.model_cfgs.n_lags,
            )
        logger.info(f"df_history shape after merge lag features: {df_history.shape} \ndf_history.columns: {df_history.columns}")
        
        # 缺失值处理
        df_history = df_history.interpolate()  # 缺失值插值填充
        df_history.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        logger.info(f"df_history shape after process NA: {df_history.shape}")
        
        # 特征排序
        if self.model_cfgs.pred_method == "univariate-multip-step-recursive":
            predict_features = [col for col in lag_features if self.model_cfgs.target in col] + weather_features + datetime_features + date_features
        else:
            predict_features = lag_features + weather_features + datetime_features + date_features
        logger.info(f"predict_features: \n{predict_features}")
        
        if exogenous_features == [] or (not self.model_cfgs.target_transform and self.model_cfgs.pred_method != "multivariate-multip-step-recursive"):
            df_history = df_history[["timeStamp"] + predict_features + [self.model_cfgs.target]]
        else:
            df_history = df_history[["timeStamp"] + predict_features + target_features]
        logger.info(f"df_history shape after feature engineering: {df_history.shape}")
        logger.info(f"df_history.head() after feature engineering: \n{df_history.head()}")
        logger.info(f"df_history.tail() after feature engineering: \n{df_history.tail()}")
        
        # 预测特征、目标变量分割
        if self.model_cfgs.date_type is not None:
            # 工作日
            df_history_workday = copy.deepcopy(df_history[df_history["datetime_weekday"] < 5])
            logger.info(f"df_history_workday.shape: {df_history_workday.shape}")
            # 非工作日
            df_history_offday = copy.deepcopy(df_history[df_history["datetime_weekday"] >= 5])
            logger.info(f"df_history_offday.shape: {df_history_offday.shape}")
            # workday 预测特征、目标变量分割
            data_X_workday = df_history_workday[predict_features]
            if exogenous_features == [] or not self.model_cfgs.target_transform:
                data_Y_workday = df_history_workday[self.model_cfgs.target]
            else:
                data_Y_workday = df_history_workday[target_features]
            if isinstance(data_Y_workday, pd.DataFrame):
                data_Y_workday.columns = [col.replace("(t+1)", "") for col in data_Y_workday.columns]
            # offday 预测特征、目标变量分割
            data_X_offday = df_history_offday[predict_features]
            if exogenous_features == [] or not self.model_cfgs.target_transform:
                data_Y_offday = df_history_offday[self.model_cfgs.target]
            else:
                data_Y_offday = df_history_offday[target_features]
            if isinstance(data_Y_offday, pd.DataFrame):
                data_Y_offday.columns = [col.replace("(t+1)", "") for col in data_Y_offday.columns]
            return data_X_workday, data_Y_workday, data_X_offday, data_Y_offday
        else:
            # 预测特征、目标变量分割
            df_history_X = df_history[predict_features]
            if exogenous_features == [] or (not self.model_cfgs.target_transform and self.model_cfgs.pred_method != "multivariate-multip-step-recursive"):
                df_history_Y = df_history[self.model_cfgs.target]
            else:
                df_history_Y = df_history[target_features]
            if isinstance(df_history_Y, pd.DataFrame):
                df_history_Y.columns = [col.replace("(t+1)", "") for col in df_history_Y.columns]
            
            return df_history_X, df_history_Y

    def process_future_data(self):
        """
        处理输入未来数据
        """
        # 未来数据
        df_future, exogenous_features, datetime_features = self._process_data_future()
        df_future, date_features = self._process_date_future(df_future)
        df_future, weather_features = self._process_weather_future(df_future)
        # 特征工程: 滞后特征
        lag_features = []
        logger.info(f"df_load lag_features: {lag_features}")
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_future.dropna(inplace=True, ignore_index=True)
        logger.info(f"df_future shape after interpolate and dropna: {df_future.shape}")
        # 特征列表
        future_feature_list = lag_features + weather_features + datetime_features + date_features
        logger.info(f"future_feature_list: \n{future_feature_list}")
        # 截取未来数据
        df_future_copy = df_future.copy()
        if self.model_cfgs.date_type is not None:
            # 工作日
            df_future_workday = copy.deepcopy(df_future[df_future["datetime_weekday"] < 5])
            logger.info(f"df_future_workday.shape: {df_future_workday.shape}")
            # 非工作日
            df_future_offday = copy.deepcopy(df_future[df_future["datetime_weekday"] >= 5])
            logger.info(f"df_future_offday.shape: {df_future_offday.shape}")
            # workday 截取未来数据
            df_future_workday.set_index("timeStamp", inplace=True)
            data_X_future_workday = df_future_workday.iloc[-self.horizon:, ]
            data_X_future_workday = df_future_workday.loc[:, future_feature_list]
            logger.info(f"data_X_future_workday: \n{data_X_future_workday.head()} \ndata_X_future_workday.columns: \n{data_X_future_workday.columns}")
            # offday 截取未来数据
            df_future_offday.set_index("timeStamp", inplace=True)
            data_X_future_offday = df_future_offday.iloc[-self.horizon:, ]
            data_X_future_offday = df_future_offday.loc[:, future_feature_list]
            logger.info(f"data_X_future_offday: \n{data_X_future_offday.head()} \ndata_X_future_offday.columns: \n{data_X_future_offday.columns}")
            return data_X_future_workday, data_X_future_offday, df_future_copy
        else:
            df_future.set_index("timeStamp", inplace=True)
            data_X_future = df_future.iloc[-self.model_cfgs.horizon:, ]
            data_X_future = df_future.loc[:, future_feature_list]
            logger.info(f"data_X_future: \n{data_X_future.head()} \ndata_X_future.columns: \n{data_X_future.columns}")

            return data_X_future, df_future_copy

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
        pred_history = list(Y_train.iloc[-int(max(lags)):-1][self.model_cfgs.target].values)
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
                if not self.model_cfgs.target_transform:
                    new_row_df[f"{self.model_cfgs.target}_{i}"] = pred_history[-i]
                else:
                    new_row_df[f"{self.model_cfgs.target}(t-{i-1})"] = pred_history[-i]
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
            pred_history_list = np.array(pred_history[-self.model_cfgs.n_lags:]).T.flatten().tolist()
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            new_row_df.iloc[:, 0:(Y_train.shape[1]*self.model_cfgs.n_lags)] = pred_history_list
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
        test_end    = -1         + (-self.model_cfgs.horizon) * (window - 1)
        test_start  = test_end   + (-self.model_cfgs.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.model_cfgs.window_len) + 1

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, \
        test_start, test_end = self._evaluate_split_index(window)
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            Y_test = data_Y.iloc[test_start:]
        else:
            X_test = data_X.iloc[test_start:(test_end+1)]
            Y_test = data_Y.iloc[test_start:(test_end+1)]

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
        测试集预测数据
        """ 
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(start=self.start_time, end=self.now_time, freq=self.model_cfgs.freq, inclusive="left")})
        if test_end == -1:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{''}")
        else:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:(test_end+1)]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{test_end+1}")
        cv_plot_df_window["train_start"] = [cv_timestamp_df["ds"].values[train_start]] * len(Y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_df["ds"].values[test_start]] * len(Y_pred)
        cv_plot_df_window["test_end"] = [cv_timestamp_df["ds"].values[test_end]] * len(Y_pred)
        cv_plot_df_window["Y_trues"] = Y_test
        cv_plot_df_window["Y_preds"] = Y_pred
        
        return cv_plot_df_window
    
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
        if self.model_cfgs.scale:
            scaler_features_test = StandardScaler()
            X_train[feature_list] = scaler_features_test.fit_transform(X_train[feature_list])
            X_test[feature_list] = scaler_features_test.transform(X_test[feature_list])
        else:
            scaler_features_test = None
        # 模型训练
        if Y_train.shape[1] == 1:
            model = lgb.LGBMRegressor(**self.model_cfgs.model_params)
            model.fit(X_train, Y_train)
        else:
            model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_cfgs.model_params))
            # model = RegressorChain(lgb.LGBMRegressor(**self.model_cfgs.model_params))
            model.fit(X_train, Y_train)
        # 模型预测
        if self.model_cfgs.pred_method == "multip-step-directly":
            Y_pred = self.univariate_directly_forecast(
                model = model,
                X_test = X_test,
            )
        elif self.model_cfgs.pred_method == "univariate-multip-step-recursive":
            Y_pred = self.univariate_recursive_forecast(
                model = model,
                X_train = X_train_df,
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.model_cfgs.lags,
                steps = min(self.model_cfgs.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        elif self.model_cfgs.pred_method == "multivariate-multip-step-recursive":
            Y_pred = self.multivariate_recursive_forecast(
                model = model,
                X_train = X_train_df, 
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.model_cfgs.lags,
                steps = min(self.model_cfgs.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        
        return Y_pred

    def test(self, data_X: pd.DataFrame, data_Y):
        """
        交叉验证
        """
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, self.model_cfgs.n_windows + 1):
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
            if self.model_cfgs.target_transform and self.model_cfgs.target_transform_predict and self.model_cfgs.pred_method != "multivariate-multip-step-recursive":
                Y_train = Y_train.iloc[:, -1:]
                Y_train.columns = [self.model_cfgs.target]
            # 模型测试
            Y_pred = self._window_test(X_train, Y_train, X_test)
            # 多变量, 单变量直接多步预测: 目标特征转换
            if self.model_cfgs.target_transform:
                # test
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_test_ups_output = Y_test.iloc[:, 0:1].values
                Y_test            = Y_test.iloc[:, -1:].values
                Y_test = Y_test * Y_test_ups_output - Y_test_ups_output
                if not self.model_cfgs.target_transform_predict:
                    # pred
                    Y_pred = np.array(Y_pred)
                    if len(Y_pred.shape) == 1:
                        Y_pred = Y_pred.reshape(-1, 1)
                    else:
                        Y_pred = pd.DataFrame(Y_pred, columns=data_Y.columns)
                        Y_pred = Y_pred.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_test_ups_output - Y_test_ups_output
                elif self.model_cfgs.target_transform_predict:
                    # pred
                    Y_preds = pd.DataFrame(Y_pred, columns=data_Y.columns)
                    Y_pred_ups_output = Y_preds.iloc[:, 0:1].values
                    Y_pred            = Y_preds.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_pred_ups_output - Y_pred_ups_output
            elif not self.model_cfgs.target_transform and self.model_cfgs.pred_method == "multivariate-multip-step-recursive":
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
        if self.model_cfgs.scale:
            scaler_features = StandardScaler()
            final_X_train[feature_list] = scaler_features.fit_transform(final_X_train[feature_list])
        else:
            scaler_features = None
        # 模型训练
        if isinstance(final_Y_train, pd.Series):
            final_model = lgb.LGBMRegressor(**self.model_cfgs.model_params)
            final_model.fit(final_X_train, final_Y_train)
        else:
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_cfgs.model_params))
            # final_model = RegressorChain(lgb.LGBMRegressor(**self.model_cfgs.model_params))
            final_model.fit(final_X_train, final_Y_train)

        return final_model, scaler_features

    def forecast(self, model, data_X, data_Y, data_X_future, data_future, scaler_features=None):
        """
        模型预测
        """
        # 模型预测
        if len(data_X_future) > 0:
            # directly multi-step forecast
            if self.model_cfgs.pred_method == "multip-step-directly":
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
            elif self.model_cfgs.pred_method == "univariate-multip-step-recursive":
                Y_pred = self.univariate_recursive_forecast(
                    model = model,
                    X_train = data_X,
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.model_cfgs.lags,
                    steps = min(self.model_cfgs.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            elif self.model_cfgs.pred_method == "multivariate-multip-step-recursive":
                Y_pred = self.multivariate_recursive_forecast(
                    model = model,
                    X_train = data_X, 
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.model_cfgs.lags,
                    steps = min(self.model_cfgs.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            # 多变量, 单变量直接多步预测: 目标变量转换
            if self.model_cfgs.target_transform:
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_future_ups_output = Y_pred_df.iloc[:, 0:1].values
                Y_pred = Y_pred_df.iloc[:, -1:].values
                Y_pred = Y_pred * Y_future_ups_output - Y_future_ups_output
            elif not self.model_cfgs.target_transform and self.model_cfgs.pred_method == "multivariate-multip-step-recursive":
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_pred = Y_pred_df.iloc[:, -1:].values
            # 预测结果收集
            data_X_future[self.model_cfgs.target] = Y_pred
        else:
            data_X_future[self.model_cfgs.target] = np.nan
        logger.info(f"data_X_future after forecast: \n{data_X_future.head()} \ndata_X_future length after forecast: {len(data_X_future)}")
        # 输出结果处理
        # data_X_future.dropna(inplace=True, ignore_index=False)
        # logger.info(f"data_X_future after dropna: \n{data_X_future.head()} \ndata_X_future length after dropna: {len(data_X_future)}")
        if self.model_cfgs.date_type is not None:
            return data_X_future
        else:
            data_future = pd.merge(data_future, data_X_future, how="outer")
            return data_future

    def run(self):
        # ------------------------------
        # 历史数据预处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model history data preprocessing...")
        logger.info(f"{80*'='}")
        if self.model_cfgs.date_type is not None:
            data_X_workday, data_Y_workday, data_X_offday, data_Y_offday = self.process_history_data()
        else:
            data_X, data_Y = self.process_history_data()
        logger.info(f"data_X: \n{data_X.head()} \ndata_X.columns: \n{data_X.columns}")
        logger.info(f"data_Y: \n{data_Y.head()}")
        logger.info(f"Model history data preprocessing over...")
        # ------------------------------
        # 模型测试
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model testing...")
        logger.info(f"{80*'='}")
        if self.model_cfgs.date_type is not None:
            test_scores_df, cv_plot_df = self.test(data_X_workday, data_Y_workday)
            test_scores_df, cv_plot_df = self.test(data_X_offday, data_Y_offday)
        else:
            test_scores_df, cv_plot_df = self.test(data_X, data_Y)
        logger.info(f"Model testing over...")
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model training...")
        logger.info(f"{80*'='}")
        if self.model_cfgs.date_type is not None:
            model_workday, scaler_features_workday = self.train(data_X_workday, data_Y_workday)
            model_offday, scaler_features_offday = self.train(data_X_offday, data_Y_offday)
        else:
            model, scaler_features = self.train(data_X, data_Y)
        logger.info(f"Model training over...")
        # ------------------------------
        # 未来数据处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model future data preprocessing...")
        logger.info(f"{80*'='}")
        if self.model_cfgs.date_type is not None:
            data_X_future_workday, data_X_future_offday, df_future = self.process_future_data()
        else:
            data_X_future, data_future = self.process_future_data()
        logger.info(f"Model history data preprocessing over...")
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model forecasting...")
        logger.info(f"{80*'='}")
        if self.model_cfgs.date_type is not None:
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
        logger.info(f"Model forecasting result: \n{df_future}")
        logger.info(f"Model forecasting over...")

        return test_scores_df, cv_plot_df, df_future


@dataclass
class ModelConfig:
    pred_method = "multip-step-directly"          # 预测方法
    scale = False                                 # 是否进行标准化
    power_ts_feat = "date"                        # 功率数据时间戳特征名称
    date_ts_feat = "time"                         # 日期数据时间戳特征名称
    weather_ts_feat = "time"                      # 天气数据时间戳特征名称
    target = "暖通功率"                            # 预测目标变量名称
    target_transform = False                      # 预测目标是否需要转换
    target_transform_predict = None               # 预测目标的转换特征是否需要预测
    freq = "5min"                                 # 数据频率
    lags = []                                     # 特征滞后数列表
    n_lags = len(lags)                            # 特征滞后数个数(1,2,...)
    n_per_day = 24 * 12                           # 每天样本数量
    history_days = 30                             # 历史数据天数
    predict_days = 1                              # 预测未来1天的功率
    window_days = 30                              # 滑动窗口天数
    horizon = predict_days * n_per_day            # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    n_windows = history_days - (window_days - 1)  # 测试滑动窗口数量, >=1, 1: 单个窗口
    window_len = window_days * n_per_day if n_windows > 1 else history_days * n_per_day   # 测试窗口数据长度(训练+测试)
    date_type = None                                                                      # 日期类型，用于区分工作日，非工作日
    now_time = datetime.datetime(2025, 5, 19, 0, 0, 0).replace(tzinfo=None, minute=0, second=0, microsecond=0)  
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
    # ------------------------------
    # model configs
    # ------------------------------
    model_cfgs = ModelConfig()
    logger.info(f"\nhistory data: {model_cfgs.time_range['start_time']} ~ {model_cfgs.time_range['now_time']} \
                  \npredict data: {model_cfgs.time_range['now_time']} ~ {model_cfgs.time_range['future_time']}")
    # ------------------------------
    # get data
    # ------------------------------
    # 实例化
    from data_provider.data_load_pue import DataLoad
    data_loader = DataLoad(data_object = "PUE")
    # data：A1-module 1
    power_df_a11 = data_loader.load_data(building_id=1, module_id=1)
    logger.info(f"power_df_a11: \n{power_df_a11}")
    # 暖通变压器功率
    power_df_hvac_a11 = power_df_a11[["date", "暖通功率"]]
    logger.info(f"power_df_hvac_a11: \n{power_df_hvac_a11}")
    # PUE single-feature
    power_df_pue_a11_univariate = power_df_a11[["date", "UPS输出功率", "PUE_变压器"]]
    logger.info(f"power_df_pue_a11_univariate: \n{power_df_pue_a11_univariate}")
    # PUE multi-features
    power_df_pue_a11 = power_df_a11[["date", "UPS输出功率", "室外温度_avg", "室外湿度_avg", "PUE_变压器"]]
    logger.info(f"power_df_pue_a11: \n{power_df_pue_a11}")
    # input_data
    input_data = {
        "df_history": power_df_hvac_a11,
        "df_date": None,
        "df_weather": None,
        "df_future": None,
        "df_date_future": None,
        "df_weather_future": None,
    }
    # input_data = {
    #     "df_history": power_df_pue_a11_univariate,
    #     "df_date": None,
    #     "df_weather": None,
    #     "df_future": None,
    #     "df_date_future": None,
    #     "df_weather_future": None,
    # }
    # input_data = {
    #     "df_history": power_df_pue_a11,
    #     "df_date": None,
    #     "df_weather": None,
    #     "df_future": None,
    #     "df_date_future": None,
    #     "df_weather_future": None,
    # }
    # ------------------------------
    # 模型测试
    # ------------------------------
    model_ins = Model(
        model_cfgs = model_cfgs,
        history_data = input_data["df_history"],
        df_date=input_data["df_date"],
        df_weather=input_data["df_weather"],
        future_data = input_data["df_future"],
        df_date_future=input_data["df_date_future"],
        df_weather_future=input_data["df_weather_future"],
    )
    test_scores_df, cv_plot_df, pred_df = model_ins.run()

if __name__ == "__main__":
    main()
