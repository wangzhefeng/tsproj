# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_fusion_stacking.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-20
# * Version     : 1.0.032021
# * Description : description
# * Link        : https://zhuanlan.zhihu.com/p/26890738
# *               https://zhuanlan.zhihu.com/p/48262572
# *               https://github.com/denotepython/my_kaggle/blob/master/titanic/python/%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%E4%B9%8B%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B.py
# *               https://www.kaggle.com/code/arthurtok/introduction-to-ensembling-stacking-in-python/notebook
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
import warnings
warnings.filterwarnings("ignore")
from typing import Dict

import numpy as np
import pandas as pd
# models
# first layer models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
# second layer model
from xgboost.sklearn import XGBClassifier
# data split
from sklearn.model_selection import (
    train_test_split,
    KFold, 
    cross_validate, 
    GridSearchCV
)
# evaluation
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model:
    
    def __init__(self, model_name: str, params: Dict = {}, random_state: int = 0):
        model_dict = {
            "logit": LogisticRegression,
            "rf": RandomForestClassifier,
            "ada": AdaBoostClassifier,
            "gbdt": GradientBoostingClassifier,
            "xgb": XGBClassifier,
        }
        params["random_state"] = random_state
        self.model_ins = model_dict[model_name](**params)
 
    def train(self, x_train, y_train):
        self.model = self.model_ins.fit(x_train, y_train)

    def predict(self, X):
        prediction = self.model.predict(X)

        return prediction


def get_oof_prediction(model, X_train, y_train, X_test, n_folds: int = 5, random_state: int = 0):
    """
    Out-Of-Fold prediction: 训练单模型分别得到下一层训练集和测试集的一列

    Args:
        clf (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        n_folds (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    # train and test samples
    n_train = X_train.shape[0]  # 890
    n_test = X_test.shape[0]  # 418
    # k-fold split
    kf = KFold(n_splits = n_folds, shuffle=False, random_state=None)

    # out-of-fold predictions
    oof_train = np.zeros((n_train,))  # 890*1
    oof_test = np.zeros((n_test,))  # 418*1
    oof_test_skf = np.empty((n_folds, n_test))  # 418*n_folds
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        # data split
        kf_X_train = X_train[train_index]  # 712*7
        kf_y_train = y_train[train_index]  # 712*1
        kf_X_test = X_train[test_index]  # 178*7
        # kf_y_test = y_train[test_index]  # 178*1
        # model training
        model.train(kf_X_train, kf_y_train)
        # model predict
        oof_train[test_index] = model.predict(kf_X_test)
        oof_test_skf[i, :] = model.predict(X_test)
    # print(f"oof_test_skf.shape: {oof_test_skf.shape}")
    # calc oof test mean
    oof_test[:] = oof_test_skf.mean(axis = 0)
    # result
    oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)
    print(f"oof_train length: {oof_train.shape}")
    print(f"oof_test length: {oof_test.shape}")

    return oof_train, oof_test




# 测试代码 main 函数
def main(): 
    # random state
    seed = 0

    # data load
    train_csv_data = pd.read_csv("./model_fusion/data/train.csv", header=0, index_col = 0) 
    test_csv_data = pd.read_csv("./model_fusion/data/test.csv", header=0, index_col = 0)
    train_csv_data = train_csv_data.ffill()
    train_csv_data = train_csv_data.bfill()
    test_csv_data = test_csv_data.ffill()
    test_csv_data = test_csv_data.bfill()
    # with pd.option_context("display.max_columns", None):
    print(f"train_csv_data.head(): \n{train_csv_data.head()}")
    print(f"train_csv_data.shape: {train_csv_data.shape}")
    print(f"test_csv_data.shape: {test_csv_data.shape}")

    # data preprocess
    X_train = train_csv_data.loc[2:, ["Pclass", "Age", "SibSp", "Parch", "Fare"]].values
    y_train = train_csv_data.loc[2:, "Survived"].values.reshape(-1, 1)
    X_test = test_csv_data[["Pclass", "Age", "SibSp", "Parch", "Fare"]].values
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")

    # data split
    train_X, test_X, train_y, test_y = train_test_split(
        X_train, 
        y_train, 
        train_size = 0.7, 
        random_state = seed
    )
    print(f"train_X.shape: {train_X.shape}")
    print(f"train_y.shape: {train_y.shape}")
    print(f"test_X.shape: {test_X.shape}")
    print(f"test_y.shape: {test_y.shape}")

    # ------------------------------
    # 第一层模型
    # ------------------------------
    # params
    logit_params = {}
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 70,
        'max_depth': 7,
        'max_features' : 'sqrt',
        'min_samples_split' : 100
    }
    ada_params = {
        'n_estimators': 40,
        'learning_rate' : 0.3
    }
    gbdt_params = {
        'n_estimators': 110,
        'learning_rate' : 0.1,
        'min_samples_leaf': 20
    }
    # models
    logit = Model(model_name = "logit")
    rf = Model(model_name = "rf", params = rf_params, random_state=seed)
    ada = Model(model_name = "ada", params = ada_params, random_state=seed)
    gbdt = Model(model_name = "gbdt", params = gbdt_params, random_state=seed)
    # 用不同的模型来得到新的 train 和 test 预测结果. 把这些结果作为新的特征
    logit_oof_train, logit_oof_test = get_oof_prediction(
        model = logit,
        X_train = train_X, y_train = train_y, X_test = test_X, 
        n_folds = 5
    )
    rf_oof_train, rf_oof_test = get_oof_prediction(
        model = rf,
        X_train = train_X, y_train = train_y, X_test = test_X, 
        n_folds = 5
    )
    ada_oof_train, ada_oof_test = get_oof_prediction(
        model = ada,
        X_train = train_X, y_train = train_y, X_test = test_X, 
        n_folds = 5
    )
    gbdt_oof_train, gbdt_oof_test = get_oof_prediction(
        model = gbdt,
        X_train = train_X, y_train = train_y, X_test = test_X,
        n_folds = 5
    )
    # 得到了第二层模型的训练集特征集合 x_train，训练集标签列 y_train，测试集特征集合 x_test, 测试集标签列 y_test
    X_train_stack = np.concatenate((
        logit_oof_train,
        rf_oof_train, 
        ada_oof_train, 
        gbdt_oof_train,
        # xgb_oof_train,
    ), axis = 1)
    X_test_stack = np.concatenate((
        logit_oof_test,
        rf_oof_test, 
        ada_oof_test, 
        gbdt_oof_test,
        # xgb_oof_test
    ), axis = 1)
    print(f"X_train_stack.shape: {X_train_stack.shape}")
    print(f"X_test_stack.shape: {X_test_stack.shape}")
 
    # ------------------------------
    # 第二层模型
    # ------------------------------
    # model training
    # xgbm = XGBClassifier(
    #     learning_rate = 0.95, 
    #     n_estimators = 16000, 
    #     max_depth = 4,
    #     min_child_weight = 2,
    #     gamma = 1,
    #     subsample = 0.8,
    #     colsample_bytree = 0.8,
    #     objective = 'binary:logistic',
    #     nthread = -1,
    #     scale_pos_weight = 1
    # )
    # xgbm.fit(X_train_stack, y_train)

    # # model predict
    # predictions = xgbm.predict(X_test_stack)
    # predictions_proba = xgbm.predict_proba(X_test_stack)
    # roc_auc_score(test_y, predictions_proba[:, 1])

if __name__ == "__main__":
    main()
