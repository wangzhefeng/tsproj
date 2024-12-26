# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-26
# * Version     : 1.0.122621
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

# metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def evaluate(Y_test, Y_pred):
    """
    模型评估

    Args:
        Y_test (_type_): _description_
        Y_pred (_type_): _description_

    Returns:
        _type_: _description_
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
    
    return eval_scores




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
