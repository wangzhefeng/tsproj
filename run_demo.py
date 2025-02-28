import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from utils.log_util import logger


def generate_time_series_data(n_samples=1000):
    """生成模拟时间序列数据"""
    np.random.seed(42)
    time = np.arange(n_samples)
    values = np.sin(0.1 * time) + 0.1 * np.random.randn(n_samples)
    return pd.DataFrame({'time': time, 'value': values})


def add_lag_features(df, lags):
    """添加滞后特征"""
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df.dropna(inplace=True)
    return df


def prepare_data(df, train_size=0.8):
    """准备训练和测试数据"""
    X = df.drop(columns=['value', 'time'])
    y = df['value']
    split_idx = int(len(df) * train_size)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_lightgbm(X_train, y_train, params, num_boost_round=100):
    """训练LightGBM模型"""
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=num_boost_round)
    return model


def recursive_forecast(model, initial_features, steps):
    """递归多步预测"""
    predictions = []
    current_features = initial_features.copy()

    for _ in range(steps):
        next_pred = model.predict(current_features.reshape(1, -1))
        predictions.append(next_pred[0])
        current_features = np.roll(current_features, shift=-1)
        current_features[-1] = next_pred
        logger.info(f"predictions: \n{predictions}")

    return predictions


def plot_results(y_train, y_test, predictions, future_steps):
    """可视化结果"""
    plt.figure(figsize=(10, 6))

    # 绘制训练集
    plt.plot(np.arange(len(y_train)), y_train, label='Training Data', color='blue')

    # 绘制测试集
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Data', color='green')

    # 绘制预测结果
    plt.plot(np.arange(len(y_train), len(y_train) + future_steps), predictions, label='Predictions', color='red', linestyle='--')

    plt.axvline(x=len(y_train), color='gray', linestyle='--', label='Train/Test Split')
    plt.legend()
    plt.title('Recursive Multi-step Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


def main():
    # 生成数据
    df = generate_time_series_data(n_samples=1000)
    logger.info(f"df: \n{df}")

    # 添加滞后特征
    lags = [1, 2, 3, 4, 5]
    df = add_lag_features(df, lags)
    logger.info(f"df: \n{df}")

    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.8)
    logger.info(f"X_train: \n{X_train}")
    logger.info(f"y_train: \n{y_train}")
    logger.info(f"X_test: \n{X_test}")
    logger.info(f"y_test: \n{y_test}")

    # 定义模型参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    # 训练模型
    model = train_lightgbm(X_train, y_train, params, num_boost_round=100)

    # 递归多步预测
    initial_features = X_train.iloc[-1].values  # 使用训练集的最后一条样本
    logger.info(f"initial_features: \n{initial_features}")
    future_steps = 10
    predictions = recursive_forecast(model, initial_features, future_steps)

    # 可视化结果
    plot_results(y_train, y_test, predictions, future_steps)

if __name__ == '__main__':
    main()
