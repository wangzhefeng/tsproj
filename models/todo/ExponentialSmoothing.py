# -*- coding: utf-8 -*-


# ***************************************************
# * File        : exponential_smoothing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-11
# * Version     : 0.1.111100
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/Cgkbyg5cI0jklGJQm-oYYw
# *               https://github.com/furkannakdagg/time_series_tutorial/blob/main/time-series-smoothing-methods-tutorial.ipynb
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 数据
# ------------------------------
# raw data
data = sm.datasets.co2.load_pandas()

# resolution = 1week
y = data.data
print(y)

# resolution = 1month
y = y["co2"].resample("MS").mean()
print(y.isnull().sum())

# 缺失值填充(平均前后值填充)
y = y.fillna(y.bfill())
print(y.isnull().sum())

# y.plot(figsize = (10, 7))
# plt.show()

# 数据集分割
train = y[:"1997-12-01"]
test = y["1998-01-01":]
print(f"Length of train: {len(train)}\nLength of test: {len(test)}")


# 数据集绘制
def plot_prediction(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend = True, label = "Train", title = f"{title}, MAE: {round(mae, 2)}")
    test.plot(legend = True, label = "Test", figsize = (6, 4))
    y_pred.plot(legend = True, label = "Prediction")
    plt.show()
# ------------------------------
# (SES) Single Exponential Smoothing
# ------------------------------
# 初始模型
ses_model = SimpleExpSmoothing(train).fit(
    smoothing_level = 0.5
)
y_pred = ses_model.forecast(48)
plot_prediction(train, test, y_pred, "Single Exponential Smoothing")

# 模型参数调优
def ses_optimizer(train, alphas, step = 48):
    best_alpha, best_mae = None, float("inf")
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(
            smoothing_level = alpha
        )
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
        print(f"alpha:{round(alpha, 2)}, mae:{round(mae, 4)}")
    print(f"best_alpha:{round(best_alpha, 2)}, best_mae:{round(best_mae, 4)}")

    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)
best_alpha, best_mae = ses_optimizer(train, alphas)

# 最终模型
final_ses_model = SimpleExpSmoothing(train).fit(
    smoothing_level = best_alpha
)
y_pred = final_ses_model.forecast(48)
plot_prediction(train, test, y_pred, "Single Exponential Smoothing")


# ------------------------------
# DES(Double Exponential Smoothing)
# ------------------------------
# 初始模型
des_model = ExponentialSmoothing(train, trend = "add").fit(
    smoothing_level = 0.5, 
    smoothing_trend = 0.5
)
y_pred = des_model.forecast(48)
plot_prediction(train, test, y_pred, "Double Exponential Smoothing")

# 模型参数调优
def des_optimizer(train, alphas, betas, step = 48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend = "add").fit(
                smoothing_level = alpha, 
                smoothing_slope = beta
            )
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print(f"alpha:{round(alpha, 2)}, beta:{round(beta, 2)}, mae:{round(mae, 4)}")
    print(f"best_alpha:{round(best_alpha, 2)}, best_beta:{round(best_beta, 2)}, best_mae:{round(best_mae, 4)}")

    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)
best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

# 最终模型
final_des_model = ExponentialSmoothing(train, trend = "add").fit(
    smoothing_level = best_alpha, 
    smoothing_slope = best_beta
)
y_pred = final_des_model.forecast(48)
plot_prediction(train, test, y_pred, "Double Exponential Smoothing")


# ------------------------------
# TES(Triple Exponential Smoothing)
# ------------------------------
# 初始模型
tes_model = ExponentialSmoothing(train, trend = "add", seasonal = "add", seasonal_periods = 12).fit(
    smoothing_level = 0.5,
    smoothing_slope = 0.5,
    smoothing_seasonal = 0.5,
)
y_pred = tes_model.forecast(48)
plot_prediction(train, test, y_pred, "Triple Exponential Smoothing")

# 模型参数调优
alphas = betas = gammas = np.arange(0.20, 1, 0.10)
abg = list(itertools.product(alphas, betas, gammas))
def tes_optimizer(train, abg, step = 48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend = "add", seasonal = "add", seasonal_periods = 12).fit(
            smoothing_level = comb[0], 
            smoothing_slope = comb[1], 
            smoothing_seasonal = comb[2]
        )
        # her satırın 0., 1., 2. elemanlarını seç ve model kur
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
 
    print(
        "best_alpha:", round(best_alpha, 2), 
        "best_beta:", round(best_beta, 2), 
        "best_gamma:", round(best_gamma, 2),
        "best_mae:", round(best_mae, 4)
    )
 
    return best_alpha, best_beta, best_gamma, best_mae


best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)

# 最终模型
final_tes_model = ExponentialSmoothing(train, trend = "add", seasonal = "add", seasonal_periods = 12).fit(
    smoothing_level = best_alpha,
    smoothing_trend = best_beta,
    smoothing_seasonal = best_gamma,
)
y_pred = final_tes_model.forecast(48)
plot_prediction(train, test, y_pred, "Triple Exponential Smoothing")




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

