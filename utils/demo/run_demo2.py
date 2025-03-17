# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_demo2.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-07
# * Version     : 1.0.030710
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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def generate_time_series_data(n_samples=1000, start_date='2020-01-01'):
    """
    生成模拟时间序列数据
    """
    np.random.seed(42)
    date_range = pd.date_range(start=start_date, periods=n_samples, freq='D')
    values = np.sin(0.1 * np.arange(n_samples)) + 0.1 * np.random.randn(n_samples)

    return pd.DataFrame({'date': date_range, 'value': values})

def add_lag_features(df, lags):
    """
    添加滞后特征
    """
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df.dropna(inplace=True)

    return df


def add_datetime_features(df):
    """
    添加日期时间特征
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday  # 星期几（0-6，0表示星期一）

    return df


def prepare_data(df, train_size=0.8):
    """
    准备训练和测试数据
    """
    X = df.drop(columns=['value', 'date'])
    print(f"X: \n{X} X.columns: {X.columns}")
    y = df['value']
    split_idx = int(len(df) * train_size)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def train_lightgbm(X_train, y_train):
    """
    训练LightGBM模型
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    model = lgb.train(params, train_data, num_boost_round=100)

    return model


def recursive_forecast(model, initial_features, initial_date, steps):
    """
    递归多步预测
    """
    predictions = []
    current_features = initial_features.copy()
    print(f"current_features 1: {current_features}")
    current_date = initial_date

    for _ in range(steps):
        # 预测下一步
        next_pred = model.predict(current_features.reshape(1, -1))
        predictions.append(next_pred[0])

        # 更新特征：将预测值作为新的滞后特征
        current_features = np.roll(current_features, shift=1)
        current_features[0] = next_pred
        print(f"predict step: {next_pred}")
        print(f"current_features 2: {current_features}")

        # 更新日期时间特征
        current_date += timedelta(days=1)
        current_features[-4] = current_date.year  # 更新年
        print(f"current_features 3: {current_features}")
        current_features[-3] = current_date.month  # 更新月
        print(f"current_features 4: {current_features}")
        current_features[-2] = current_date.day  # 更新日
        print(f"current_features 5: {current_features}")
        current_features[-1] = current_date.weekday()  # 更新星期
        print(f"current_features 6: {current_features}")
        break

    return predictions


def plot_results(y_train, y_test, predictions, future_steps):
    """
    可视化结果
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_train)), y_train, label='Training Data', color='blue')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Data', color='green')
    plt.plot(np.arange(len(y_train), len(y_train) + future_steps), predictions, label='Predictions', color='red', linestyle='--')
    plt.axvline(x=len(y_train), color='gray', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title('Recursive Multi-step Forecasting with Datetime Features')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


def main():
    # 生成数据
    df = generate_time_series_data(n_samples=1000, start_date='2020-01-01')

    # 添加滞后特征
    lags = [1, 2, 3, 4, 5]  # 滞后1到5步
    df = add_lag_features(df, lags)

    # 添加日期时间特征
    df = add_datetime_features(df)
    print(f"df: \n{df.head()} \ndf.columns: {df.columns}")

    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.8)

    # 训练模型
    model = train_lightgbm(X_train, y_train)

    # 递归多步预测
    initial_features = X_train.iloc[-1].values  # 使用训练集的最后一条样本
    print(initial_features)
    initial_date = df.iloc[len(X_train) - 1]['date']  # 最后一条样本的日期
    print(initial_date)
    future_steps = 50  # 预测未来50步
    predictions = recursive_forecast(model, initial_features, initial_date, future_steps)

    # 可视化结果
    # plot_results(y_train, y_test, predictions, future_steps)

if __name__ == "__main__":
    main()
