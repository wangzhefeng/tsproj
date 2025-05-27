import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

from utils.plot_results import plot_forecast_results


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


def train_lightgbm(X_train, y_train):
    """训练LightGBM模型"""
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


def train_arima(y_train):
    """训练ARIMA模型"""
    model = ARIMA(y_train, order=(5, 1, 0))  # ARIMA(p=5, d=1, q=0)
    model_fit = model.fit()
    return model_fit


def recursive_forecast(model, initial_features, steps, model_type='lightgbm'):
    """递归多步预测"""
    predictions = []
    current_features = initial_features.copy()

    for _ in range(steps):
        if model_type == 'lightgbm':
            next_pred = model.predict(current_features.reshape(1, -1))
        elif model_type == 'arima':
            next_pred = model.forecast(steps=1)
            model = model.append([next_pred], refit=False)
        predictions.append(next_pred[0])
        current_features = np.roll(current_features, shift=-1)
        current_features[-1] = next_pred

    return predictions


def weighted_average(predictions_1, predictions_2, weight_1=0.7, weight_2=0.3):
    """加权平均集成"""
    return weight_1 * np.array(predictions_1) + weight_2 * np.array(predictions_2)




def main():
    # 生成数据
    df = generate_time_series_data(n_samples=1000)

    # 添加滞后特征
    lags = [1, 2, 3, 4, 5]  # 滞后1到5步
    df = add_lag_features(df, lags)

    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.8)

    # 训练LightGBM模型
    lgb_model = train_lightgbm(X_train, y_train)

    # 训练ARIMA模型
    arima_model = train_arima(y_train)

    # 递归多步预测（LightGBM）
    initial_features = X_train.iloc[-1].values  # 使用训练集的最后一条样本
    future_steps = 50  # 预测未来50步
    lgb_predictions = recursive_forecast(lgb_model, initial_features, future_steps, model_type='lightgbm')

    # 递归多步预测（ARIMA）
    arima_predictions = recursive_forecast(arima_model, None, future_steps, model_type='arima')

    # 加权平均集成
    ensemble_predictions = weighted_average(lgb_predictions, arima_predictions, weight_1=0.7, weight_2=0.3)

    # 可视化结果
    plot_forecast_results(y_train, y_test, ensemble_predictions, future_steps)

if __name__ == '__main__':
    main()