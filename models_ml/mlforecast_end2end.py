# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mlforecast_end2end.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091102
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
import random
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from utilsforecast.plotting import plot_series
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from numba import njit
from window_ops.rolling import rolling_mean
import lightgbm as lgb
from mlforecast.lgb_cv import LightGBMCV
from utilsforecast.losses import rmse
from mlforecast.lag_transforms import ExponentiallyWeightedMean

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data download and filter: M4 hourly
async def load_download_data():
    from datasetsforecast.m4 import M4
    await M4.async_download()
    df, *_ = M4.load('data', 'Hourly')
    # data filter
    uids = df['unique_id'].unique()
    random.seed(0)
    sample_uids = random.choices(uids, k=4)
    df = df[df['unique_id'].isin(sample_uids)].reset_index(drop=True)
    df['ds'] = df['ds'].astype('int64')
    
    return df


def load_data():
    # data read and filer
    df = pd.read_parquet("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet")
    print(df["unique_id"].value_counts())
    # data filter
    uids = df['unique_id'].unique()
    random.seed(0)
    sample_uids = random.choices(uids, k = 4)
    df = df[df['unique_id'].isin(sample_uids)].reset_index(drop = True)
    df['ds'] = df['ds'].astype('int64')
    print(df["unique_id"].value_counts())
    return df

df = load_data()

# ------------------------------
# EDA
# ------------------------------
# 模型的周期为 1day = 24hour
fig = plot_series(df, max_insample_length = 24*24)
plt.show();

# subtract seasonality using difference(remove seasonality)
fcst = MLForecast(
    models = [],  # not interested in modeling
    freq = 1,  # series have integer timestamps, so just add 1 in every timestamp
    target_transforms=[Differences([24])],
)
prep = fcst.preprocess(df)
print(prep)

# after subtacted the lag 24 from each value
fig = plot_series(prep)
plt.show();

'''
# ------------------------------
# feature engine
# ------------------------------
# add lags features
# fcst = MLForecast(
#     models = [],
#     freq = 1,
#     lags = [1, 24],
#     target_transforms=[Differences([24])],
# )
# prep = fcst.preprocess(df)
# print(prep)
# lags features 与预测变量的相关性
# print(prep.drop(columns=["unique_id", "ds"]).corr()["y"])

# Lag transforms
# @njit
# def rolling_mean_48(x):
#     return rolling_mean(x, window_size = 48)

# fcst = MLForecast(
#     models = [],
#     freq = 1,
#     target_transforms = [Differences([24])],
#     lag_transforms = {
#         1: [ExpandingMean()],
#         24: [RollingMean(window_size = 48), rolling_mean_48]
#     },
# )
# prep = fcst.preprocess(df)
# print(prep)

# Date features
def hour_index(times):
    return times % 24

# fcst = MLForecast(
#     models = [],
#     freq = 1,
#     target_transforms = [Differences([24])],
#     date_features = [hour_index],
# )
# fcst.preprocess(df)

# target transformations
# fcst = MLForecast(
#     models = [],
#     freq = 1,
#     lags = [1],
#     target_transforms = [LocalStandardScaler()],
# )
# fcst.preprocess(df)

# ------------------------------
# model training
# ------------------------------
# models
## custom Navie
from sklearn.base import BaseEstimator
class Navie(BaseEstimator):
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return X["lag1"]

## lightgbm
lgb_params = {
    "verbosity": -1,
    "num_leaves": 512,
}

# models build
fcst = MLForecast(
    models = {
        "avg": lgb.LGBMRegressor(**lgb_params),
        "q75": lgb.LGBMRegressor(**lgb_params, objective = "quantile", alpha = 0.75),
        "q25": lgb.LGBMRegressor(**lgb_params, objective = "quantile", alpha = 0.25)
    },
    freq = 1,
    target_transforms = [Differences([24])],
    lags = [1, 24],
    lag_transforms = {
        1: [ExpandingMean()],
        24: [RollingMean(window_size = 48)],
    },
    date_features = [hour_index],
)
fcst.fit(df)

# ------------------------------
# model forecasting
# ------------------------------
preds = fcst.predict(h = 48)
print(preds)
fig = plot_series(df, preds, max_insample_length = 24 * 7)

# ------------------------------
# model save and load
# ------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    # save
    save_dir = Path(tmpdir) / "mlforecast"
    fcst.save(save_dir)
    # load
    fcst2 = MLForecast.load(save_dir)
    preds2 = fcst2.predict(h = 48)
    pd.testing.assert_frame_equal(preds, preds2)

# ------------------------------
# update series's value
# ------------------------------
fcst = MLForecast(
    models = [Navie()],
    freq = 1,
    lags = [1, 2, 3],
)
fcst.fit(df)
fcst.predict(1)

new_values = pd.DataFrame({
    "unique_id": ["H196", "H256"],
    "ds": [1009, 1009],
    "y": [17.0, 14.0],
})
fcst.update(new_values)
preds = fcst.predict(h = 1)
print(preds)

# ------------------------------
# model evaluation
# ------------------------------
# cross validation
# ------------------
fcst = MLForecast(
    models = lgb.LGBMRegressor(**lgb_params),
    freq = 1,
    target_transforms = [Differences([24])],
    lags = [1, 24],
    lag_transforms = {
        "1": [ExpandingMean()],
        "24": [RollingMean(window_size = 48)],
    },
    date_features = [hour_index],
)
cv_result = fcst.cross_validation(
    df,
    n_windows = 4,  # number of models to train/split to perform
    h = 48,  # length of the validation set in each window
)
print(cv_result)

# 计算每个分割的 RMSE
def evaluate_cv(df):
    return rmse(df, models = ["LGBMRegressor"], id_col = "cutoff").set_index("cutoff")

split_rmse = evaluate_cv(cv_result)
print(split_rmse)
# 计算所有分割的 RMSE 均值
print(split_rmse.mean())


# 方法尝试: lag_transforms: ExponentiallyWeightedMean
fcst = MLForecast(
    models = lgb.LGBMRegressor(**lgb_params),
    freq = 1,
    lags = [1, 24],
    lag_transforms = {
        1: [ExponentiallyWeightedMean(alpha = 0.5)],
        24: [RollingMean(window_size = 48)],
    },
    date_features = [hour_index],
)
cv_result2 = fcst.cross_validation(
    df, 
    n_windows = 4,
    h = 48,
)
evaluate_cv(cv_result2).mean()




# LightGBMCV
# ------------------
cv = LightGBMCV(
    freq = 1,
    target_transforms = [Differences([24])],
    lags = [1, 24],
    lag_transforms = {
        "1": [ExpandingMean()],
        "24": [RollingMean(window_size = 48)],
    },
    date_features = [hour_index],
    num_threads = 2,
)
cv_hist = cv.fit(
    df,
    n_windows = 4,
    h = 48,
    params = lgb_params,
    eval_every = 5,
    early_stopping_evals = 5,
    compute_cv_preds = True,
)
# out-of-fold predictions using the best iteration
print(cv.cv_preds_)
fig = plot_series(forecasts_df = cv.cv_preds_.drop(columns = "window"))
# final model
final_fcst = MLForecast.from_cv(cv)
final_fcst.fit(df)
preds = final_fcst.predict(h = 48)
fig = plot_series(df, preds, max_insample_length = 24 * 24)
'''


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
