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
import datetime
from typing import Dict, List

# tools
import numpy as np
import pandas as pd
# models
import lightgbm as lgb
# metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr

from utils.feature_engine import (
    extend_datetime_stamp_feature,
    extend_lag_feature,
    extend_date_type_feature,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    def __init__(self, 
                 model_cfgs: Dict,
                 history_data: pd.DataFrame, 
                 future_data: pd.DataFrame) -> None:
        self.model_cfgs = model_cfgs
        self.history_data = history_data
        self.future_data = future_data

    def _get_history_data(self):
        """
        历史数据处理
        """
        start_time = self.model_cfgs["time_range"]["start_time"]
        now_time = self.model_cfgs["time_range"]["now_time"]
        freq = self.model_cfgs["freq"]
        # 构造时间戳完整的历史数据：生成以 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_history = pd.DataFrame({"ds": pd.date_range(start=start_time, end=now_time, freq=freq, inclusive="left")})
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
                    # 将数据转换为字符串类型
                    df[col] = df[col].apply(lambda x: float(x))
                    # 将原始数据映射到时间戳完整的 df_history 中, 特征包括[ds, y, exogenous_features]
                    df_history[col] = df_history["ds"].map(df.set_index("ds")[col])
        
        return df_history

    def _get_future_data(self):
        """
        未来数据处理
        """
        now_time = self.model_cfgs["time_range"]["now_time"]
        future_time = self.model_cfgs["time_range"]["future_time"]
        freq = self.model_cfgs["freq"]
        # 构造时间戳完整的历史数据：生成未来 freq 间隔的时间序列，创建 pandas.DataFrame 并添加 ds 列
        df_future = pd.DataFrame({"ds": pd.date_range(start=now_time, end=future_time, freq=freq, inclusive="left")})
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
            target=self.model_cfgs["target"], 
            group_col=None, 
            numLags=self.model_cfgs["lags"],
            numHorizon=0, 
            dropna=True,
        )
        df_history_lags_feats = [
            col for col in df_history_lags 
            if col.startswith(f"{self.model_cfgs['target']}(")
        ]
        
        return df_history_lags, df_history_lags_feats
    
    def _get_future_lag_features(self, df_history, df_future):
        """
        滞后特征构造
        """
        # params
        now_time = self.model_cfgs["time_range"]["now_time"]
        future_time = self.model_cfgs["time_range"]["future_time"]
        freq = self.model_cfgs["freq"]
        # 特征构造
        df_future_lags = extend_lag_feature(
            df=df_history, 
            target=self.model_cfgs["target"], 
            group_col=None, 
            numLags=0, 
            numHorizon=self.model_cfgs["lags"], 
            dropna=False,
        )
        # 筛选样本
        df_future_lags = df_future_lags.iloc[-self.model_cfgs["horizon"]:, ]
        # 时间戳修改为未来时间戳
        df_future_lags["ds"] = pd.date_range(start=now_time, end=future_time, freq=freq, inclusive="left")
        # 滞后特征合并
        df_future_lags_feats = [col for col in df_future_lags if col.startswith(f"{self.model_cfgs['target']}(")]
        for col in df_future_lags_feats:
            df_future[col] = df_future["ds"].map(df_future_lags.set_index("ds")[col])
        
        return df_future, df_future_lags_feats

    def process_input_history_data(self):
        """
        处理输入历史数据
        """
        # ------------------------------
        # 数据预处理
        # ------------------------------
        df_history = self._get_history_data()
        # ------------------------------
        # 特征工程
        # ------------------------------
        # 目标特征
        # target_feats = [target]
        # 时间戳特征
        # timestamp_feats = ["ds"]
        # 外生特征
        exogenous_features = [col for col in df_history.columns if col != "ds" and col != self.model_cfgs["target"]]
        # 日期时间特征
        df_history, datetime_features = self._get_datetime_features(
            df_history, 
            datetime_features = [
                # 'minute', 
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
        # ------------------------------
        # 预测特征、目标变量分割
        # ------------------------------
        # 特征筛选
        predict_features = datetime_features + exogenous_features + lag_features
        df_history_X = df_history[predict_features]
        df_history_Y = df_history[self.model_cfgs["target"]]
        
        return df_history_X, df_history_Y

    def process_input_future_data(self):
        """
        处理输入未来数据
        """
        # ------------------------------
        # 数据预处理
        # ------------------------------
        df_history = self._get_history_data()
        df_future = self._get_future_data()
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
        exogenous_features = [col for col in df_future.columns if col != "ds" and col != self.model_cfgs["target"]]
        # 日期时间特征
        df_future, datetime_features = self._get_datetime_features(
            df_future,
            datetime_features = [
                # 'minute', 
                'hour', 'day', 
                'weekday', 'week', 'day_of_week', 'week_of_year', 
                'month', 'days_in_month', 'quarter', 
                'day_of_year', 'year'
            ]
        )
        # 滞后特征
        df_future, lag_features = self._get_future_lag_features(df_history, df_future)
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
        valid_end   = -1        + (-self.model_cfgs["horizon"]) * (window - 1)
        valid_start = valid_end + (-self.model_cfgs["horizon"]) + 1
        train_end   = valid_start
        train_start = valid_end + (-self.model_cfgs["data_length"]) + 1

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

    def train(self, X_train, Y_train):
        """
        模型训练
        """
        model_params = self.model_cfgs["model_params"]
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
        start_time = self.model_cfgs["time_range"]["start_time"]
        now_time = self.model_cfgs["time_range"]["now_time"]
        freq = self.model_cfgs["freq"]
        # 数据分割指标
        train_start, train_end, valid_start, valid_end = self._cv_split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(start=start_time, end=now_time, freq=freq, inclusive="left")})
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
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, n_windows + 1):
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self.cv_split(data_X, data_Y, window)
            logger.info(f"length of X_train: {len(X_train)}, length of Y_train: {len(Y_train)}")
            logger.info(f"length of X_test: {len(X_test)}, length of Y_test: {len(Y_test)}")
            # 模型训练
            model = self.train(X_train, Y_train)
            # 模型验证
            Y_pred = self.valid(model, X_test)
            # 模型评价
            eval_scores = self.evaluate(Y_test, Y_pred, window)
            test_scores_df = pd.concat([test_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self.evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        logger.info(f"cross validation scores: \n{test_scores_df}") 
        # 模型评价指标数据处理
        test_scores_df = test_scores_df.mean()
        test_scores_df = test_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
        
        return test_scores_df, cv_plot_df

    def run(self):
        # ------------------------------
        # 数据处理
        # ------------------------------
        logger.info(f"history data process...")
        df_history_X, df_history_Y = self.process_input_history_data()
        # ------------------------------
        # 模型训练、验证
        # ------------------------------
        logger.info(f"model training...")
        # 模型训练、评价
        test_scores_df, cv_plot_df = self.cross_validation(
            df_history_X, 
            df_history_Y,
            n_windows = self.model_cfgs["n_windows"],
        )
        logger.info(f"cross validation average scores: \n{test_scores_df}")
        # ------------------------------
        # 模型重新训练
        # ------------------------------
        final_model = self.train(X_train=df_history_X, Y_train=df_history_Y)
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"future data process...")
        df_future_X = self.process_input_future_data()
        
        logger.info(f"model predict...")
        if self.model_cfgs["pred_method"] == "multip-step-directly":  # 模型单步预测
            pred_df = self.predict(final_model, df_future_X)
        elif self.model_cfgs["pred_method"] == "multip-step-recursion":  # 模型多步递归预测
            pred_df = np.array([])
            for step in range(self.model_cfgs["horizon"]):
                logger.info(f'step {step} predict...')
                df_future_x_row = df_future_X.iloc[step, ].values
                df_future_x_row = np.delete(df_future_x_row, np.where(np.isnan(df_future_x_row)))
                X_future = np.concatenate([df_future_x_row, pred_df])
                pred_value = self.predict(final_model, X_future.reshape(1, -1))
                pred_df = np.concatenate([pred_df, pred_value])
        
        logger.info(f"model predict result: \n{pred_df}")
        logger.info(f"model predict over...")

        return pred_df, test_scores_df, cv_plot_df

    def multip_step_directly(self):
        # input info
        pred_method = "multip-step-directly"                                           # 预测方法
        freq = "15min"                                                                 # 数据频率
        target = "load"                                                                # 预测目标变量名称
        lags = 0 if pred_method == "multip-step-directly" else 96                      # 滞后特征构建
        # TODO 是否设置正确
        n_windows = 1                                                                  # cross validation 窗口数量
        history_days = 30                                                              # 历史数据天数
        predict_days = 1                                                               # 预测未来1天的功率
        data_length = 15 * 96 if n_windows > 1 else history_days * 24 * 4              # 训练数据长度
        horizon = predict_days * 24 * 4                                                # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
        now = datetime.datetime(2025, 2, 4, 0, 0, 0)                                   # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
        future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻
        
        # model params
        model_cfgs = {
            "pred_method": pred_method,
            "time_range": {
                "start_time": start_time,
                "now_time": now_time,
                "future_time": future_time,
            },
            "data_length": data_length,
            "horizon": horizon,
            "freq": freq,
            "lags": lags,
            "target": target,
            "n_windows": n_windows,
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
        }
        # model
        model_ins = Model(
            model_cfgs=model_cfgs,
            history_data=self.history_data,
            future_data=self.future_data,
        )
        # model running
        pred_df, eval_scores, _ = model_ins.run()

    def mutlip_step_recursion(self):
        # input info
        pred_method = "multip-step-recursion"                                          # 预测方法
        freq = "15min"                                                                 # 数据频率
        target = "load"                                                                # 预测目标变量名称
        lags = 0 if pred_method == "multip-step-directly" else 96                      # 滞后特征构建
        # TODO 是否设置正确
        n_windows = 16                                                                 # cross validation 窗口数量
        history_days = 30                                                              # 历史数据天数
        predict_days = 1                                                               # 预测未来1天的功率
        data_length = 15 * 96 if n_windows > 1 else history_days * 24 * 4              # 训练数据长度
        horizon = predict_days * 24 * 4                                                # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
        now = datetime.datetime(2025, 2, 4, 0, 0, 0)                                   # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
        future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻
        # model params
        model_cfgs = {
            "pred_method": pred_method,
            "time_range": {
                "start_time": start_time,
                "now_time": now_time,
                "future_time": future_time,
            },
            "data_length": data_length,
            "horizon": horizon,
            "freq": freq,
            "lags": lags,
            "target": target,
            "n_windows": n_windows,
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
        }
        # model
        model_ins = Model(
            model_cfgs=model_cfgs,
            history_data=self.history_data,
            future_data=self.future_data,
        )
        # model running
        pred_df, eval_scores, _ = model_ins.run()




# 测试代码 main 函数
def main():
    pass
    
if __name__ == "__main__":
    main()
