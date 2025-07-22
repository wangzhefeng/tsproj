# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LightGBM_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description : description
# * Link        : https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247487120&idx=1&sn=9783b3d4f75f2c3282452815c26d7a49&chksm=96c42355a1b3aa43392107567654bac5a15e91ef89201a77be1047a80429d9c43e1bd2774200&scene=178&cur_album_id=1364202321906941952&search_click_id=#rd
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

    def __init__(self, args: Dict) -> None:
        self.args = args
 
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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
 