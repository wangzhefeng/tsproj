# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metrics.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-02
# * Version     : 0.1.110215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import Union, List

import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import pearsonr

from utils.dtw_metric import accelerated_dtw

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MAE_v2(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MAE(mean absolute error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)

    return mean_absolute_error(true, pred)


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def MSE_v2(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MSE(mean squared error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)
    
    return mean_squared_error(true, pred)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MAPE_v2(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MAPE(mean absolute percentage error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)

    return mean_absolute_percentage_error(true, pred)


def MAPE_v3(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MAPE(mean absolute percentage error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)
    
    return np.mean(np.abs((true - pred) / true)) * 100


def Accuracy(pred, true):
    true, pred = np.array(true), np.array(pred)
    
    return 1 - mean_absolute_percentage_error(true, pred)


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def DTW(preds, trues):
    dtw_list = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(preds.shape[0]):
        x = preds[i].reshape(-1,1)
        y = trues[i].reshape(-1,1)
        if i % 100 == 0:
            print("calculating dtw iter:", i)
        d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
        dtw_list.append(d)
    dtw = np.array(dtw_list).mean()
    
    return dtw


def R2(pred, true):
    true, pred = np.array(true), np.array(pred)
    return r2_score(true, pred)


def R2_V2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)


def metric(pred, true):
    rse = RSE(pred, true)
    # corr = CORR(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    accuracy = Accuracy(pred, true)
    mspe = MSPE(pred, true)
    dtw = DTW(pred, true)
    r2 = R2(pred, true)

    return rse, mae, mse, rmse, mape, accuracy, mspe, dtw, r2


def cal_accuracy(y_pred, y_true):
    """
    计算准确率
    """
    return np.mean(y_pred == y_true)


def evaluate(pred, true):
    """
    模型评估
    """
    # 计算模型的性能指标
    if true.mean() == 0:
        true = true.apply(lambda x: x + 0.01)
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    accuracy = 1 - mean_absolute_percentage_error(true, pred)
    # correlation, p_value = pearsonr(true, pred)
    eval_scores = {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "accuracy": accuracy,
        # "correlation": correlation,
    }
    
    return eval_scores




# 测试代码 main 函数
def main():
    np.random.seed(0)
    y_true = np.random.rand(10)
    y_pred = np.random.rand(10)
    print(y_true)
    print(y_pred)
    rse, mae, mse, rmse, mape, accuracy, mspe, dtw, r2 = metric(y_pred, y_true)
    print(f"rse: {rse}\nmae: {mae}\nmse: {mse}\nrmse: {rmse}\nmape: {mape}\
        \naccuracy: {accuracy}\nmspe: {mspe}\ndtw: {dtw}\nr2: {r2}")

if __name__ == "__main__":
    main()
