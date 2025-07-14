# -*- coding: utf-8 -*-

# ***************************************************
# * File        : prophet_fit.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091118
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/fpaV8KixV3lJePF1lMTiRA
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
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from models.ts_feature.TimeFeatures import time_static_features
from models.utils import metrics

plt.style.use("fivethirtyeight")
color_pal = [
    "#F8766D", "#D39200", "#93AA00",
    "#00BA38", "#00C19F", "#00B9E3",
    "#619CFF", "#DB72FB"
]
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data
# ------------------------------
# data load
df = pd.read_csv(
    "E:/projects/timeseries_forecasting/tsproj/models/dataset/PJME_hourly.csv", 
    index_col = [0], 
    parse_dates = [0]
)
df.columns = ["y"]
df.index.name = "ds"
print(df.head())
print(df.tail())
print(df.shape)

# data visual
# df.plot(style = '.', figsize = (15, 5), color = color_pal[1], title = 'PJM East')
# plt.show();


# ------------------------------
# 特征工程
# ------------------------------
X = time_static_features(
    df, 
    dt_is_index = True, 
    features = [
        "hour", "dayofweek", "quarter", "month",
        "year", "dayofyear", "dayofmonth", "weekofyear"
    ],
)
y = df["y"]
df = pd.concat([X, y], axis = 1)
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df.head())
    print(df.tail())


# ------------------------------
# EDA
# ------------------------------
# sns.pairplot(
#     df.dropna(),
#     hue = "hour",
#     x_vars = ["hour", "dayofweek", "year", "weekofyear"],
#     y_vars = 'y',
#     kind = 'scatter',
#     height = 5,
#     plot_kws = {
#         'alpha': 0.05, 
#         'linewidth': 0
#     },
#     palette = 'husl'
# )
# plt.suptitle('Power Use MW by Hour, Day of Week, Year and Week of Year')
# plt.show();

# 结论：
# TODO


# ------------------------------
# 数据分割
# ------------------------------
# train, test split
split_date = "2015-01-01"
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()
print(df_train.shape)
print(df_test.shape)

# data visual
# df_visual = df_test \
#     .rename(columns = {'y': 'TEST SET'}) \
#     .join(df_train.rename(columns = {'y': 'TRAIN SET'}), how = 'outer', lsuffix="test", rsuffix="train")
# df_visual[["TEST SET", "TRAIN SET"]].plot(figsize = (15, 5), title = 'PJM East', style = '.')
# plt.show();


# ------------------------------
# 模型构建
# ------------------------------
"""
# 使用默认参数
model = Prophet() 
# 趋势项相关设置
model = Prophet(growth = 'logistic')  # 默认是 linear
model = Prophet(changepoints = ['2020-01-01', '2023-01-01'])  #手工设置 changepoints
model = Prophet(n_changepoints = 25) 
model = Prophet(changepoint_range = 0.8)
model = Prophet(changepoint_prior_scale = 0.05)  # 越大越容易转变趋势

# 周期项相关设置
model = Prophet(yearly_seasonality = 'auto')
model = Prophet(weekly_seasonality = True)
model = Prophet(daily_seasonality = False)
model = Prophet(seasonality_mode = 'multiplicative')  # 默认是 additive
model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5, prior_scale = 0.1)  # 手工添加周期项

# 节假日项相关设置
model.add_country_holidays(country_name = 'AU')
model = Prophet(holidays = dfholidays, holidays_mode = 'multiplicative') 
model.add_regressor('temperature')  # 使用温度特征作为一个回归项因子，需要在训练集和测试集中都知道
"""
# holiday
cal = calendar()
df["ds"] = df.index.date
df["is_holiday"] = df["ds"].isin(d.date() for d in cal.holidays())
df_holidays = df.loc[df["is_holiday"] == True]
df_holidays["holiday"] = "USFederalHoliday"
df_holidays = df_holidays.drop(["y", "is_holiday"], axis = 1)
df_holidays["ds"] = pd.to_datetime(df_holidays["ds"])

# model build
model = Prophet(holidays = df_holidays)

# ------------------------------
# 模型训练
# ------------------------------
model.fit(df_train.reset_index(), iter = 10000, show_console = True)

'''
# ------------------------------
# 模型预测
# ------------------------------
# model predict
df_test_fcst = model.predict(df = df_test.reset_index())

# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(df_test.index, df_test['y'], color = 'r')  # 真实值
fig = model.plot(df_test_fcst, ax = ax)  # 预测值以及范围
plt.show();

# ------------------------------
# 模型评估
# ------------------------------
MSE = metrics.mse(y_true = df_test['y'], y_pred = df_test_fcst['yhat'])
MAE = metrics.mae(y_true = df_test['y'], y_pred = df_test_fcst['yhat'])
MAPE = metrics.mape(y_true = df_test['y'], y_pred = df_test_fcst['yhat'])
print(f'mse = {MSE}')
print(f'mae = {MAE}')
print(f'mape = {MAPE}')

# 真实值和预测值差异比较
ax = df_test_fcst.set_index('ds')['yhat'].plot(figsize = (15, 5), lw = 0, style = '.')
df_test['y'].plot(ax = ax, style='.', lw = 0, alpha = 0.2)
plt.legend(['Forecast','Actual'])
plt.title('Forecast vs Actuals')
plt.show();

# ------------------------------
# 
# ------------------------------
# 加法模型结构
# display(df_test_fcst.tail())
cols = [x for x in df_test_fcst.columns if 'lower' not in x and 'upper' not in x] 
print(cols)

# yhat = trend+additive_terms
print((df_test_fcst['yhat'] - df_test_fcst.eval('trend+additive_terms')).sum())

# additive_terms = daily+weekly+yearly+multiplicative_terms+USFederalHoliday
print((df_test_fcst['additive_terms'] - df_test_fcst.eval('daily+weekly+yearly+multiplicative_terms+USFederalHoliday')).sum())

# model explaination
fig = model.plot_components(df_test_fcst)

# ------------------------------
# 模型保存
# ------------------------------
model_path = ""
with open(model_path, "w") as md:
    json.dump(model_to_json(model), md)

with open(model_path, "r") as md:
    model_loaded = model_from_json(json.load(md))
'''



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
