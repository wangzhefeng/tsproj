# -*- coding: utf-8 -*-

# ***************************************************
# * File        : power_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-11
# * Version     : 1.0.031122
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
import random
import tempfile
from pathlib import Path

# data utils
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utilsforecast.plotting import plot_series
# data
from mlforecast.utils import generate_daily_series
# main module
from mlforecast import MLForecast
# transforms
from mlforecast.lag_transforms import (
    ExpandingMean, 
    RollingMean, 
    ExponentiallyWeightedMean,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler
from numba import njit
from window_ops.rolling import rolling_mean
# model
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import lightgbm as lgb
from mlforecast.lgb_cv import LightGBMCV
from sklearn.base import BaseEstimator
# metric
from utilsforecast.losses import mse, rmse, mae, mape, smape
from sklearn.metrics import r2_score

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
df_history = pd.read_csv()
df_future = pd.read_csv()

series = generate_daily_series(
    n_series = 20,
    max_length = 100,
    n_static_features = 1,
    static_as_categorical = False,
    with_trend = True,
)
print(series.head())
print()
print(series.shape)
print()
print(series["unique_id"].value_counts())


# ------------------------------
# model
# ------------------------------
lgbm_params = {
    "verbosity": -1,
    "num_leaves": 512,
    "random_state": 0,
}
fcst = MLForecast(
    models = {
        "lgbm_avg": lgb.LGBMRegressor(*lgbm_params),
        "lgbm_q75": lgb.LGBMRegressor(*lgbm_params, objective="quantile", alpha=0.75),
        "lgbm_q25": lgb.LGBMRegressor(*lgbm_params, objective="quantile", alpha=0.25),
        "linear": LinearRegression(),
        "lgbm_cv": LightGBMCV(), 
    },
    freq = "15min",
    lags = [1, 2, 4, 5],
    lag_transforms = {
        1: [ExpandingMean()],
        1: [ExponentiallyWeightedMean(alpha=0.5)],
        1: [RollingMean(window_size = 28)],
    },
    date_features = [],
    target_transforms = [
        Differences([1]),
        LocalStandardScaler(),
    ],
)


# ------------------------------
# feature engine
# ------------------------------
# fcst = MLForecast(
#     models = [],
#     freq = "15min",
#     lags = [12],
#     target_transforms=[Differences([12])],
# )

# date features
def hour_index(times):
    return times % 24

# lag rolling mean
@njit
def rolling_mean_48(x):
    return rolling_mean(x, window_size = 48)

prep_data = fcst.preprocess(df_future)
with pd.option_context("display.max_columns", None):
    print(prep_data.head(20))
    print()
    print(prep_data.shape)
    print()
    print(prep_data["unique_id"].value_counts())

prep_data[["lag1", "lag2", "lag3", "lag4", "y"]].corr()


# ------------------------------
# model training
# ------------------------------
fcst.fit()


# ------------------------------
# model evaluation: cross validation
# ------------------------------
# cross validation
cv_result = fcst.cross_validation(
    df_history,
    n_windows = 4,
    h = 96,
)
print(cv_result)

# plot cross validation forecasting
fig = plot_series(forecasts_df = cv_result.drop(columns = "cutoff"))
print(fig)

# 计算每个分割的 RMSE
def evaluate_cv(df):
    return rmse(df, models = ["LGBMRegressor"], id_col = "cutoff").set_index("cutoff")

split_rmse = evaluate_cv(cv_result)
print(f"split_rmse:\n{split_rmse}")

# 计算所有分割的 RMSE 均值
print(f"\nsplit_rmse_mean:\n{split_rmse.mean()}")

# -----------------------------
# cross validation
cv = LightGBMCV(
    freq = "15min", 
    lags = [1, 24],
    lag_transforms = {
        1: [ExpandingMean()],
        24: [RollingMean(window_size = 48)],
    },
    date_features = [hour_index],
    target_transforms = [
        Differences([24]),
        LocalStandardScaler(),
    ],
    num_threads = 0,
)
cv_hist = cv.fit(
    df_history,
    n_windows = 4,
    h = 96,
    params = lgbm_params,
    eval_every = 5,
    early_stopping_evals = 5,
    compute_cv_preds = True,
)

# out-of-fold predictions using the best iteration
preds = cv.cv_preds_
print(preds)

fig = plot_series(forecasts_df = preds.drop(columns = "window"))
print(fig)

# ------------------------------
# final model
# ------------------------------
# final model
final_fcst = MLForecast.from_cv(cv)
final_fcst.fit(df_history)

# ------------------------------
# model forecasting
# ------------------------------
# version 1
preds = fcst.predict(h = 96, X_df = df_future)
print(preds)

# version 2
preds = final_fcst.predict(h = 48)
print(preds)

# forecasting visual
fig = plot_series(df_history, preds)
print(fig)


# ------------------------------
# model save and load
# ------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    model_dir = Path(tmpdir) / "mlforecast"
    # save
    fcst.save(model_dir)
    # load
    fcst2 = MLForecast.load(model_dir)
    # predict
    preds2 = fcst2.predict(h = 48)
    pd.testing.assert_frame_equal(preds, preds2)






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
