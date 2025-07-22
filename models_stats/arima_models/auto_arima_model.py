# -*- coding: utf-8 -*-

# ***************************************************
# * File        : auto_arima_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-10
# * Version     : 1.0.091021
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
from typing import Tuple

import pmdarima as pm
# 开发环境信息
print(pm.show_versions())

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_load(data_path: str):
    """
    模型记载

    Args:
        data_path (str): _description_
    """
    pass


def data_split(data):
    """
    模型分割

    Args:
        data (_type_): _description_
    """
    pass


def select_model(series):
    """
    模型选择

    Args:
        series (_type_): _description_

    Returns:
        _type_: _description_
    """
    stepwise_fit = pm.auto_arima(
        y = series,
        X = None,
        stationary = False,
        max_order = 5,
        start_p = 1,
        max_p = 3,
        d = 1,
        max_d = 2,
        start_q = 1,
        max_q = 3,
        seasonal = True,
        start_P = 2,
        max_P = 12,
        D = 1,
        max_D = 12,
        start_Q = 2,
        max_Q = 12,
        m = 10,
        trace = True,
        error_action = "ignore",
        suppress_warnings = True,  # 收敛信息
        stepwise = True,
        information_criterion = "aic",
        alpha = 0.05,
        test = "kpss",
    )
    print(stepwise_fit.summary())
    order = ()
    seasonal_order = ()
    
    return stepwise_fit, order, seasonal_order


def build_model(order: Tuple, seasonal_order: Tuple, train, test):
    """
    模型构建

    Args:
        order (Tuple): _description_
        seasonal_order (Tuple): _description_
        train (_type_): _description_
        test (_type_): _description_

    Returns:
        _type_: _description_
    """
    arima = pm.ARIMA(order = order, seasonal_order = seasonal_order)
    arima.fit(train)
    # 更新模型
    arima.update(test)

    return arima


def model_predict(model, test):
    """
    模型预测

    Args:
        model (_type_): _description_
        test (_type_): _description_
    """
    preds = model.predict(n_periods = test.shape[0])

    return preds


def save_model(model, model_path: str, method: str = "pickle"):
    """
    模型保存

    Args:
        model (_type_): _description_
        model_path (str): _description_
        method (str, optional): _description_. Defaults to "pickle".
    """
    assert method in ["pickle", "joblib"], "method must be one of 'pickle', 'joblib'"

    if method == "pickle":
        import pickle
        # pickle serialize model
        with open(model_path, "wb") as pkl:
            pickle.dump(model, pkl)
    else:
        import joblib
        # joblib serialize model
        with open(model_path, "wb") as pkl:
            joblib.dump(model, pkl)


def load_model(model_path: str, method: str = "pickle"):
    """
    模型加载

    Args:
        model_path (str): _description_
        method (str, optional): _description_. Defaults to "pickle".

    Returns:
        _type_: _description_
    """
    assert method in ["pickle", "joblib"], "method must be one of 'pickle', 'joblib'"
    if method == "pickle":
        import pickle
        # pickle load model
        with open(model_path, "rb") as pkl:
            model = pickle.load(pkl)
    else:
        import joblib
        # pickle load model
        with open(model_path, "rb") as pkl:
            model = joblib.load(pkl)
    
    return model


def load_model_predict(model_path: str, method: str = "pickle", len_pred: int = 5):
    """
    模型加载

    Args:
        model_path (str): _description_
        method (str, optional): _description_. Defaults to "pickle".

    Returns:
        _type_: _description_
    """
    assert method in ["pickle", "joblib"], "method must be one of 'pickle', 'joblib'"
    if method == "pickle":
        import pickle
        # pickle load model
        with open(model_path, "rb") as pkl:
            preds = pickle.load(pkl).predict(n_periods = len_pred)
    else:
        import joblib
        # pickle load model
        with open(model_path, "rb") as pkl:
            preds = model = joblib.load(pkl).predict(n_periods = len_pred)
    
    return preds


def run_model():
    # 模型选择
    
    # 模型构建
    # 模型保存
    # 模型预测
    pass




# 测试代码 main 函数
def main():
    # data
    import numpy as np
    from pmdarima.datasets import load_wineind
    wineind = load_wineind().astype(np.float32)

if __name__ == "__main__":
    main()
