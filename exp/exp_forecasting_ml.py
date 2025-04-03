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
    extend_datetime_features,
    extend_date_type_features,
    extend_weather_features,
    extend_future_weather_features,
    extend_lag_features,
)
# utils
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    def __init__(self, args: Dict, history_data, future_data) -> None:
        self.args = args
        # datetime index
        self.train_start_time_str = self.args.train_start_time.strftime("%Y%m%d")
        self.train_end_time_str = self.args.train_end_time.strftime("%Y%m%d")
        self.forecast_start_time_str = self.args.forecast_start_time.strftime("%Y%m%d")
        self.forecast_end_time_str = self.args.forecast_end_time.strftime("%Y%m%d")
        # data
        self.history_data = history_data
        self.future_data = future_data

    def load_csv_data(self):
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
    def __calc_features_corr(self, df_history, train_features):
        """
        分析预测特征与目标特征的相关性
        """
        features_corr = df_history[train_features + ['load']].corr()
    
        return features_corr

    def preprocessing_history_data(self, input_data: Dict = None):
        """
        历史数据预处理
        """
        # 历史数据格式
        df_history = pd.DataFrame({
            "ds": pd.date_range(
                self.args.train_start_time, 
                self.args.train_end_time, 
                freq = self.args.freq,
                inclusive = "left",
            ),
            # "unique_id": None,
            # "y": None,
        })
        # 特征工程：目标时间序列特征
        df_series_history = self.__process_df_timestamp(
            df = input_data["df_series_history"] if input_data is not None else self.history_data, 
            date_col = self.args.df_target_ds_col, 
            new_date_col = "ds",
        )  # count_data_time
        df_history = self.__process_target_series(
            df_history,
            df_series_history, 
            col_numeric = self.args.target_series_numeric_features, 
            col_categorical = self.args.target_series_categorical_features,
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
        # df_history_X, df_history_Y = df_history[train_features], df_history[self.args.target]

        return df_history, train_features

    def preprocessing_future_data(self, input_data):
        """
        处理未来数据
        """
        # 未来数据格式
        df_future = pd.DataFrame({
            "ds": pd.date_range(
                self.args.forecast_start_time,
                self.args.forecast_end_time,
                freq = self.args.freq,
                inclusive = "left",
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
        # df_future_X = df_future[future_features]

        return df_future, future_features

    def __split_index(self, window: int):
        """
        Cross-Validation 数据分割索引构建
        """
        test_end    = -1         + (-self.args.horizon) * (window - 1)
        test_start  = test_end   + (-self.args.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.args.data_length) + 1

        return train_start, train_end, test_start, test_end

    def __cross_validation_split(self, data_X, data_Y, window: int):
        """
        Cross-Validation 训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self.__split_index(window)
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            y_test = data_Y.iloc[test_start:]
            logger.info(f"split indexes:: \ntrain_start:train_end: {train_start}:{train_end} \
                \ntest_start:test_end: {test_start}:{''}")
        else:
            X_test = data_X.iloc[test_start:(test_end+1)]
            y_test = data_Y.iloc[test_start:(test_end+1)]
            logger.info(f"split indexes:: \ntrain_start:train_end: {train_start}:{train_end}, \
                \ntest_start:test_end: {test_start}:{test_end+1}")
        logger.info(f"length of X_train: {len(X_train)}, length of y_train: {len(y_train)}")
        logger.info(f"length of X_test: {len(X_test)}, length of y_test: {len(y_test)}")

        return X_train, y_train, X_test, y_test

    def __multi_step_directly_forecast(self, model, X_future, scaler_features):
        """
        模型预测
        """
        # 标准化
        if self.args.scale:
            X_future = scaler_features.transform(X_future)
        # 模型预测
        if len(X_future) > 0:
            y_pred = model.predict(X_future)
            return y_pred
        else:
            y_pred = []
            logger.info(f"X_future length is 0!")
            return y_pred

    def __multi_step_recursive_forecast(self, model, X_future, df_history, scaler_features = None):
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
        self.args.horizon = min(self.args.horizon, len(X_future))
        for step in range(self.args.horizon):
            # 初始预测特征
            if self.args.scale:
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            next_pred = next_pred.reshape(-1, 1)
            # 更新 pred_history
            pred_history.append(next_pred[0])
            
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1].copy()
            # date, weather features update
            for future_feature in X_future.columns:
                if future_feature != "ds":
                    new_row_df[future_feature] = X_future.iloc[step][future_feature]
            # lag features update
            for i in self.args.lags:
                if i > len(pred_history): break
                new_row_df[f"lag_{i}"] = pred_history[-i]
            
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, pd.DataFrame([new_row_df])], axis = 0, ignore_index=True)

            # 收集预测结果
            predictions.append(next_pred[0])
        
        return predictions
    
    @staticmethod
    def __model_evaluate(y_test, y_pred, window: int = 1):
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
            "accuracy": 1 - mean_absolute_percentage_error(y_test, y_pred) 
        }
        test_scores_df = pd.DataFrame(test_scores, index=[window])
        logger.info(f"test_scores_df: \n{test_scores_df}")
        
        return test_scores_df
    
    def __process_cv_prediction(self, y_test, y_pred, window: int):
        """
        测试集预测数据
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self.__split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(
            self.args.train_start_time, 
            self.args.train_end_time, 
            self.args.freq,
            inclusive = "left",
        )})
        cv_plot_df_window["ds"] = cv_timestamp_df[test_start:] \
            if test_end == -1 \
            else cv_timestamp_df[test_start:(test_end+1)]
        cv_plot_df_window["train_start"] = [cv_timestamp_df["ds"].values[train_start]] * len(y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_df["ds"].values[test_start]] * len(y_pred)
        cv_plot_df_window["test_end"] = [cv_timestamp_df["ds"].values[test_end]] * len(y_pred)
        cv_plot_df_window["Y_tures"] = y_test
        cv_plot_df_window["Y_preds"] = y_pred
        
        return cv_plot_df_window 
    
    def __plot_cv_prediction(self, cv_plot_df, df_history, df_predict):
        # history ture data
        history_data = df_history.dropna()
        history_data.set_index("ds", inplace=True)
        history_data = history_data.loc[history_data.index >= min(cv_plot_df.index), ]
        # future predict data
        predict_data = df_predict.dropna()
        predict_data.set_index("ds", inplace=True)
        # plot
        for col_hist_pred, col_cv in zip(
            history_data.columns,
            [col for col in cv_plot_df.columns if col not in ["train_start", "cutoff", "valid_end"]]
        ):
            plt.figure(figsize = (15, 5))
            plt.plot(history_data[col_hist_pred], label = "实际值", linewidth = "1.5")
            plt.plot(cv_plot_df[col_cv], label = "预测值", linewidth = "1.5", ls = "--")
            plt.plot(predict_data[col_hist_pred], label = "预测值", linewidth = "1.5")
            for cutoff in cv_plot_df["cutoff"].unique():
                plt.axvline(x = cutoff, color = "red", ls = ":", linewidth = "1.0")
            plt.title(f"预测测试值对比时序图")
            plt.xlabel(f"Date [{self.args.freq}]")
            plt.ylabel("Value [kW]")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show();
    
    def train(self, X_train, y_train, model_params):
        """
        模型训练
        """
        # 归一化/标准化
        if self.args.scale == "standard":
            scaler_features= StandardScaler()
            X_train = scaler_features.fit_transform(X_train)
        elif self.args.scale == "minmax":
            scaler_features= MinMaxScaler()
            X_train = scaler_features.fit_transform(X_train)
        elif self.args.scale is None:
            scaler_features = None
        # 模型训练
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)
        # 特征重要性排序
        if self.args.plot_importance:
            lgb.plot_importance(
                model, 
                importance_type="gain",  # gain, split
                figsize=(7,6), 
                title="LightGBM Feature Importance (Gain)"
            )
            plt.show();
        
        return model, scaler_features
    
    def forecast(self, model, X_future, X_history, y_history, scaler_features):
        """
        模型验证
        """
        if len(X_future) > 0:
            if self.args.pred_method == "multi-step-directly":
                y_pred = self.__multi_step_directly_forecast(
                    model = model,
                    X_future = X_future,
                    scaler_features = scaler_features,
                )
            elif self.args.pred_method == "multi-step-recursion":
                y_pred = self.__multi_step_recursive_forecast(
                    model = model,
                    X_future = X_future,
                    df_history = pd.concat([X_history, y_history], axis=1),
                    scaler_features = scaler_features,
                )
            else:  # TODO
                y_pred = []
            logger.info(f"y_pred: {y_pred} \ny_pred length: {len(y_pred)}")
            return y_pred
        else:
            y_pred = []
            logger.info(f"X_test length is 0!")
            return y_pred
    
    def cross_validation(self, data_X, data_Y, model_params, n_windows: int, drop_last_window: bool = True):
        """
        训练数据交叉验证
        """
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, n_windows + 1):  # 1 ~ n_windows-1
            # 数据分割: 训练集、测试集
            X_train, y_train, \
            X_test, y_test = self.__cross_validation_split(
                data_X = data_X, 
                data_Y = data_Y, 
                window = window,
            )
            # TODO
            if len(X_train) == 0:
                break
            # 模型训练
            model, scaler_features = self.train(
                X_train = X_train, 
                y_train = y_train, 
                model_params = model_params,
            )
            # 模型验证
            y_pred = self.forecast(
                model = model, 
                X_future = X_test, 
                X_history = X_train, 
                y_history = y_train, 
                scaler_features = scaler_features
            )
            # 模型测试指标
            test_scores_df_window = self.__model_evaluate(y_test, y_pred, window)
            test_scores_df = pd.concat([test_scores_df, test_scores_df_window], axis = 0)
            # 模型测试预测绘图数据
            cv_plot_df_window = self.__process_cv_prediction(y_test, y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        logger.info(f"cross validation scores: \n{test_scores_df}")
        # 模型评价指标数据处理
        test_scores_df_avg = test_scores_df.mean() \
            .to_frame().T \
            .reset_index(drop = True, inplace = False)
        logger.info(f"cross validation average scores: \n{test_scores_df_avg}") 

        return test_scores_df_avg, cv_plot_df
 
    # TODO
    def model_save(self, final_model):
        pass

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
        logger.info(f"load history and future data...")
        input_data = self.load_csv_data()
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"history data process...")
        df_history, train_features = self.preprocessing_history_data(input_data = input_data)
        df_history_X, df_history_Y = df_history[train_features], df_history[self.args.target]
        # ------------------------------
        # 未来数据处理
        # ------------------------------
        logger.info(f"future data process...")
        df_future, future_features = self.preprocessing_future_data(input_data = input_data)
        df_future_X = df_future[future_features]
        # ------------------------------
        # 模型选择/模型调参
        # ------------------------------
        final_model_params = None
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_test:
            logger.info(f"model testing...")
            test_scores_df, plot_df = self.cross_validation(
                data_X = df_history_X,
                data_Y = df_history_Y,
                model_params = self.args.model_params,
                n_windows = self.args.n_windows,
                drop_last_window = True,
            )
            logger.info(f"model test scores: \n{test_scores_df}")
            logger.info(f"model test plot_df: \n{plot_df}") 
            logger.info(f"model testing over...")
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.is_train:
            logger.info(f"model training...")
            # 模型训练
            final_model, scaler_features = self.train(
                X_train = df_history_X,
                y_train = df_history_Y,
                model_params = final_model_params,
            )
            # 模型保存
            self.model_save(final_model)
            logger.info(f"model saved to path: {self.args.model_save_path}")
            logger.info(f"model training over...")
        # ------------------------------
        # 模型预测
        # ------------------------------
        if self.args.is_forecast: 
            logger.info(f"model forecasting...")
            # 模型预测
            y_pred = self.forecast(
                model = final_model,
                X_future = df_future_X,
                X_history = df_history_X,
                y_history = df_history_Y,
                scaler_features = scaler_features,
            )
            # 模型输出
            y_pred = self.process_output(y_pred)
            logger.info(f"model forecast prediction: {y_pred}")
            logger.info(f"model forecasting over...")




# 测试代码 main 函数
def main():
    logger.info("=" * 50)
    logger.info("Load parameters for traning...")
    logger.info("=" * 50)
    args = {
        "history_days": 30,
        "predict_days": 5,
        "freq": "15min",
        "target": "load",
        "data_path": "./dataset/",  # input data path
        # "time_range": {
        #     "train_start_time": train_start_time,
        #     "train_end_time": train_end_time,
        #     "forecast_start_time": forecast_start_time,
        #     "forecast_end_time": forecast_end_time,
        # },
        "is_workday": True,
        "is_train": True,
        "is_test": True,
        "is_forecast": True,
        "pred_method": "multi-step-directly", 
        "lags": [1, 2, 3, 4],
        "n_windows": 15,
        "horizon": 5 * 24 * 4,  # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度 , 
        "demand_load_min_thread": 66,
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
        "result_path": f"./saved_results/predict_results/",
    }

if __name__ == "__main__":
    main()
