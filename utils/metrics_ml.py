# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metrics_ml.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-14
# * Version     : 1.0.011415
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
from typing import Union, List

import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import pearsonr

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def MSE(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MSE(mean squared error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)
    
    return mean_squared_error(true, pred)


def MAE(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MAE(mean absolute error) given true and pred
    """
    true, pred = np.array(true), np.array(pred)

    return mean_absolute_error(true, pred)


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
    pass

if __name__ == "__main__":
    main()
