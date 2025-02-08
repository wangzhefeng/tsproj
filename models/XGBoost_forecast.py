# -*- coding: utf-8 -*-

# ***************************************************
# * File        : XGBoost_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
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
import xgboost as xgb
# metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr

# log
from utils.log_util import logger
from utils.feature_engine import (
    extend_datetime_stamp_feature,
    extend_lag_feature,
    extend_date_type_feature,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:

    def __init__(self, 
                 project_name: str, 
                 model_name: str, 
                 model_cfgs: Dict,
                 history_data: pd.DataFrame, 
                 future_data: pd.DataFrame) -> None:
        self.project_name = project_name
        self.model_name = model_name
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
        valid_end   = -1        + (-self.model_cfgs["horizon"]) * window
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

    def train(self, X_train, Y_train, X_test, Y_test):
        """
        模型训练
        """
        model_params = self.model_cfgs["model_params"]
        train_xgb = xgb.DMatrix(X_train, label=Y_train)
        test_xgb = xgb.DMatrix(X_test, label=Y_test)
        model = xgb.train(
            params =  model_params,
            dtrain = train_xgb,
            evals = [
                (train_xgb, 'train'),
                (test_xgb, 'val')
            ],
            num_boost_round = 100,
            early_stopping_rounds = 10,
            verbose_eval = 5,
            xgb_model = None
        )
        
        return model
    
    def valid(self, model, X_test, Y_test):
        """
        模型验证
        """
        valid_xgb = xgb.DMatrix(X_test, label=Y_test)
        Y_pred = model.predict(valid_xgb)
        
        return Y_pred

    def predict(self, model, X_future):
        """
        模型预测
        """
        test_xgb = xgb.DMatrix(X_future)
        Y_pred = model.predict(test_xgb)
        
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
        eval_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(n_windows):
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self.cv_split(data_X, data_Y, window)
            logger.info(f"length of X_train: {len(X_train)}, length of Y_train: {len(Y_train)}")
            logger.info(f"length of X_test: {len(X_test)}, length of Y_test: {len(Y_test)}")
            # 模型训练
            model = self.train(X_train, Y_train, X_test, Y_test)
            # 模型验证
            Y_pred = self.valid(model, X_test)
            # 模型评价
            eval_scores = self.evaluate(Y_test, Y_pred, window)
            eval_scores_df = pd.concat([eval_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self.evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        
        return eval_scores_df, cv_plot_df

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
        eval_scores_df, cv_plot_df = self.cross_validation(
            df_history_X,
            df_history_Y,
            n_windows = self.model_cfgs["n_windows"],
        )
        logger.info(f"cross validation scores: \n{eval_scores_df}")
        # 模型评价指标数据处理
        eval_scores_df = eval_scores_df.mean()
        eval_scores_df = eval_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
        logger.info(f"cross validation average scores: \n{eval_scores_df}")
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
        # 模型单步预测
        if self.model_cfgs["pred_method"] == "multip-step-directly":
            pred_df = self.predict(final_model, df_future_X)
        # 模型多步递归预测
        elif self.model_cfgs["pred_method"] == "multip-step-recursion":
            pred_df = np.array([])
            for step in range(self.model_cfgs["horizon"]):
                logger.info(f'step {step} predict...')
                df_future_x_row = df_future_X.iloc[step, ].values
                df_future_x_row = np.delete(df_future_x_row, np.where(np.isnan(df_future_x_row)))
                X_future = np.concatenate([df_future_x_row, pred_df])
                pred_value = self.predict(final_model, X_future.reshape(1, -1))
                pred_df = np.concatenate([pred_df, pred_value])
        # TODO 模型多步直接预测
        elif self.model_cfgs["pred_method"] == "multip-step-directly-lags":
            pred_df = []
        logger.info(f"model predict result: \n{pred_df}")
        logger.info(f"model predict over...")
        return pred_df, eval_scores_df, cv_plot_df




# 测试代码 main 函数
def main():
    # input info
    history_days = 14
    predict_days = 1
    now = datetime.datetime(2024, 12, 1, 0, 0, 0)

    # input params 
    now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)
    future_time = now_time + datetime.timedelta(days=predict_days)
    model_cfgs = {
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": 24 *7,
        "split_length": 24,
        "freq": "1h",
        "datetime_feat": True,
        "target": "load",
        "model_params": {
            'booster': 'gbtree',  # 弱评估器
            'objective': 'reg:squarederror',
            'verbosity':2,  # 打印消息的详细程度
            'tree_method': 'auto',  # 这是xgb框架自带的训练方法，可选参数为[auto,exact,approx,hist,gpu_hist]
            'eval_metric': 'rmse',  # 评价指标
            'max_depth': 6,  # 树的深度，默认为 6，一般取值 [3,10] 越大偏差越小，方差越大，需综合考虑时间及拟合性
            'min_child_weight': 3,  # 分裂叶子节点中样本权重和的最小值，如果新分裂的节点的样本权重和小于 min_child_weight 则停止分裂，默认为 1，取值范围 [0,],当值越大时，越容易欠拟合，当值越小时，越容易过拟合
            'gamma':0,  # 别名 min_split_loss，指定节点分裂所需的最小损失很熟下降值,节点分裂时，只有损失函数的减小值大于等于gamma，节点才会分裂，gamma越大，算法越保守，取值范围为[0,] 【0,1,5】
            'subsample': 0.8,  # 训练每棵树时子采样比例，默认为 1，一般在 [0.5,1] 之间，调节该参数可以防止过拟合
            'colsample_bytree': 0.7,  # 训练每棵树时，使用特征占全部特征的比例，默认为 1，典型值为 [0.5,1]，调节该参数可以防止过拟合
            'alpha':1,  # 别名 reg_alpha，L1 正则化，在高维度的情况下，调节该参数可以加快算法的速度，增大该值将是模型更保守，一般我们做特征选择的时候会用L1正则项，
            'lambda': 2,  # L2 正则化，调节、、增大该参数可以减少过拟合，默认值为 1
            'eta': 0.3,  # 别名 learning_rate 学习率一般越小越好，只是耗时会更长
            # 'n_estimators':500,  # 基学习器的个数，越大越好，偏差越小，但耗时会增加
            # 'max_delat_step':2,  # 限制每棵树权重改变的最大步长，默认值为 0，及没有约束，如果为正值，则这个算法更加保守，通常不需要设置该参数，但是当样本十分不平衡时，对逻辑回归有帮助
            'nthread': -1,  # 有多少处理器可以使用，默认为 1，-1表示没有限制。
            # 'silent': 1,  # 默认为 0，不输出中间过程，1：输出中间过程
            'seed' : 2023,  # 随机种子
            # 'is_unbalance':True
        }
    }
    
    # input data
    history_data = None
    future_data = None

    # model
    model_instance = Model(
        project_name="aidc",
        model_name="xgb",
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    )

    # model running
    model_instance.run(cross_validation=True)
    
if __name__ == "__main__":
    main()
