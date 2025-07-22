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

# features
from feature_engineering.datetime_features import extend_datetime_features
from feature_engineering.datetype_features import extend_datetype_features
from feature_engineering.weather_featurs import (
    extend_weather_features, 
    extend_future_weather_features
)
from feature_engineering.lag_features import extend_lag_features

# utils
from utils.log_util import logger


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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
