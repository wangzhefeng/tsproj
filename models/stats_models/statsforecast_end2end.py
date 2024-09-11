# -*- coding: utf-8 -*-

# ***************************************************
# * File        : statsforecast_end2end.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

"""
Auto Forecast:
    - large collections of univariate time series
    - ARIMA, ETS, Theta, CES
Exponential Smoothing:
    - data with no clear trend or seasonality
    - SES, Holt's Winters, SSO
Benchmark models:
    - Mean, Navie, Random Walk
Intermittent or Sparse models:
    - series with very few non-zero observations
    - CROSTON, ADIDA, IMAPA
Multiple Seasonalities:
    - signals with more than one clear seasonality
    - low-frequency data
    - MSTL
Theta Models:
    - deseasonalized time series
    - Theta, DynamicTheta
"""

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston,
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive,
)
import multiprocessing as mp


os.environ["NIXTLA_ID_AS_COL"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
def load_data():
    # data read
    Y_df = pd.read_parquet("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet")
    # print(Y_df.head())
    # print(Y_df.shape)
    print(Y_df["unique_id"].value_counts())
    # data filter
    uids = Y_df["unique_id"].unique()[:10]
    Y_df = Y_df.query("unique_id in @uids")
    Y_df = Y_df.groupby("unique_id").tail(7 * 24)
    # print(Y_df.head())
    # print(Y_df.shape)
    print(Y_df["unique_id"].value_counts())

    return Y_df


Y_df = load_data()


# ------------------------------
# EDA
# ------------------------------
# StatsForecast.plot(Y_df, engine="matplotlib")
# plt.show();


# ------------------------------
# model training
# ------------------------------
models = [
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24),
]
sf = StatsForecast(
    models = models,
    freq = 1,
    fallback_model=SeasonalNaive(season_length=7),
    n_jobs = 1,
)


# ------------------------------
# model evaluation
# ------------------------------
forecasts_df = sf.forecast(df = Y_df, h = 48, level = [90])
print(forecasts_df)
sf.plot(Y_df, forecasts_df)
sf.plot(Y_df, forecasts_df, models = ["HoltWinters", "DynamicOptimizedTheta"], unique_ids = ["H10", "H105"], level = [90])

# ------------------------------
# best model
# ------------------------------






# 测试代码 main 函数
def main():
    mp.freeze_support()

if __name__ == "__main__":
    main()
