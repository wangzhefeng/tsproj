# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-04
# * Version     : 1.0.060414
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
import xgboost as xgb

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


def code_mean(data, cate_feature, real_feature):
    """
    # TODO

    Args:
        data (_type_): _description_
        cate_feature (_type_): 类别型特征， 如星期几
        real_feature (_type_): target 字段
    """
    return dict(
        data.groupby(cate_feature)[real_feature] \
            .mean()
    )


def prepareData(series, target, lag_start, lag_end, test_size: float, target_encoding=False):
    # copy of the initial dataset
    data = pd.DataFrame(series.copy()).loc[:, [target]]
    data.columns = ["y"]
    logger.info(f"data: \n{data}")
    
    # lags of series
    for i in range(lag_start, lag_end):
        data[f"lag_{i}"] = data["y"].shift(i)
    logger.info(f"data: \n{data}")
    
    # datatime features
    data.index = pd.to_datetime(data.index)
    data["hour"] = data.index.hour
    data["weekday"] = data.index.weekday
    data["is_weekend"] = data.index.weekday.isin([5, 6]) * 1
    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(data.dropna()) * (1 - test_size))
        data["weekday_average"] = list(map(
            code_mean(data[:test_index], "weekday", "y").get, data.weekday
        ))
        # drop encoded variables
        data.drop(["weekday"], axis=1, inplace=True)
    logger.info(f"data: \n{data} \ndata.columns: {data.columns}")

    # train-test split
    y = data.dropna()["y"]
    X = data.dropna().drop(["y"], axis=1)
    X = pd.get_dummies(X)
    logger.info(f"y: \n{y}")
    logger.info(f"X: \n{X} \nX.columns: \n{X.columns}")
    test_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:test_index], X.iloc[test_index:]
    y_train, y_test = y.iloc[:test_index], y.iloc[test_index:]

    return X_train, X_test, y_train, y_test




# 测试代码 main 函数
def main():
    # data path
    data_dir = Path("./dataset/ETT-small/")
    logger.info(f"data_dir: {data_dir}")

    # data
    df = pd.read_csv(
        data_dir / "ETTh1.csv",
        parse_dates=["date"],
        index_col="date"
    )
    logger.info(f"df: \n{df.head()}")
    logger.info(f"df na: \n{df.isna().sum()}")
    series = df[["OT"]]
    logger.info(f"series: \n{series}")
    
    # data split
    X_train, X_test, y_train, y_test = prepareData(
        series, 
        target="OT",
        lag_start=1, 
        lag_end=28, 
        test_size=0.1,
        target_encoding=True,
    )
    logger.info(f"X_train: \n{X_train} \nX_train.shape: {X_train.shape}")
    logger.info(f"X_test: \n{X_test} \nX_test.shape: {X_test.shape}")
    logger.info(f"y_train: \n{y_train} \ny_train.shape: {y_train.shape}")
    logger.info(f"y_test: \n{y_test} \ny_test.shape: {y_test.shape}")
    
    # data transform
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # model training
    model = xgb.XGBRegressor()
    model.fit(X_train_scaled, y_train)

    # model test
    y_pred = model.predict(X_test_scaled)
    logger.info(f"y_pred: \n{y_pred}")

    # model result visiual
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.values, label="test actual", lw=2.0)
    plt.plot(y_pred, label="test pred", lw=2.0)
    logger.info(f"y_test: \n{len(y_test)}")
    logger.info(f"y_pred: \n{len(y_pred)}")
    
    # 模型预测区间
    plot_intervals = True
    if plot_intervals:
        # data split
        tscv = TimeSeriesSplit(n_splits=5)
        tscv.split(series)
        cv = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        deviation = np.sqrt(cv.std())
        scale = 1.96
        lower = y_pred - (scale * deviation)
        upper = y_pred + (scale * deviation)
        plt.plot(lower, linestyle="--", alpha=0.2, label="lower bond")
        plt.plot(upper, linestyle="--", alpha=0.2, label="upper bond")
    
    # 异常值
    plot_anomalies = True
    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(y_test))
        anomalies[y_test < lower] = y_test[y_test < lower]
        anomalies[y_test > upper] = y_test[y_test > upper]
        plt.plot(anomalies, "o", markersize=4, color="C3", label="Anomalies")
    
    plt.title(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test.values, y_pred):.2f}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.show();

if __name__ == "__main__":
    main()
