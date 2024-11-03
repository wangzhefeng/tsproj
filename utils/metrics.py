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
from sklearn.metrics import mean_squared_error, mean_absolute_error 

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


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


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


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    # dtw = DTW(pred, true)

    return mae, mse, rmse, mape, mspe#, dtw


# ------------------------------
# addtional functions
# ------------------------------
def MSE_v2(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
    """
    Calculates MSE(mean squared error) given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return mean_squared_error(y_true, y_pred)


def MAE_v2(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
    """
    Calculates MSE(mean absolute error) given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return mean_absolute_error(y_true, y_pred)


def MAPE_v2(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
    """
    Calculates MAPE(mean absolute percentage error) given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
