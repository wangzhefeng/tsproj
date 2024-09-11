# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metrics.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-11
# * Version     : 0.1.091123
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
from typing import List, Union

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error 

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def mse(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
    """
    Calculates MSE(mean squared error) given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return mean_squared_error(y_true, y_pred)
   

def mae(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
    """
    Calculates MSE(mean absolute error) given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return mean_absolute_error(y_true, y_pred)


def mape(y_true: Union[List, np.array], y_pred: Union[List, np.array]):
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
