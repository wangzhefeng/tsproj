# -*- coding: utf-8 -*-

# ***************************************************
# * File        : forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121115
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
import time
import warnings
import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        model = self.model_dict[self.args.model].Model(self.args)
        
        return model

    def _get_data(self, flag: str):
        """
        数据集构建

        Args:
            flag (str): 任务类型, ["train", "val", "test"]

        Returns:
            _type_: Dataset, DataLoader
        """
        history_data, future_data = data_provider(self.args, flag)

        return history_data, future_data

    def train(self):
        pass
    
    def vali(self):
        pass

    def test(self, test = 0):
        pass

    def predict(self):
        pass




# 测试代码 main 函数
def main():
    # ------------------------------
    # params
    # ------------------------------
    # input info
    pred_method = "multip-step-recursion"                                          # 预测方法
    freq = "1h"                                                                    # 数据频率
    lags = 24                                                                      # 滞后特征构建
    target = "load"                                                                # 预测目标变量名称
    n_windows = 1                                                                  # cross validation 窗口数量
    history_days = 14                                                              # 历史数据天数
    predict_days = 1                                                               # 预测未来1天的功率
    data_length = 8 * 24 if n_windows > 1 else history_days * 24                   # 训练数据长度
    horizon = predict_days * 24                                                    # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
    now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
    start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
    future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻
    model = "LightGBM_forecast"
    model_params = {
        "LightGBM_forecast": {
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
        "XGBoost_forecast": {
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
        },
    }

if __name__ == "__main__":
    main()
