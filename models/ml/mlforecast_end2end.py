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
from datasetsforecast.m4 import M4
from utilsforecast.plotting import plot_series
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# ------------------------------
# data
# ------------------------------
# M4 hourly
async def load_data():
    await M4.async_download("data", group = "Hourly")
    df, *_ = M4.load("data", "Hourly")
    uids = df["unique_id"].unique()
    random.seed(0)
    sample_uids = random.choices(uids, k = 4)
    df = df[df["unique_id"].isin(sample_uids)].reset_index(drop = True)
    df["ds"] = df["ds"].astype("int64")
    
    return df

df = load_data()

# ------------------------------
# EDA
# ------------------------------
# 模型的周期为 1day = 24hour
fig = plot_series(df, max_insample_length = 24*24)

# subtract seasonality using difference(remove seasonality)
fcst = MLForecast(
    model = [],  # not interested in modeling
    freq = 1,  # series have integer timestamps, so just add 1 in every timestamp
    target_transforms=[Differences([24])],
)
prep = fcst.preprocess(df)
print(prep)

# after subtacted the lag 24 from each value
fig = plot_series(prep)


# ------------------------------
# feature engine
# ------------------------------
# add lags features
fcst = MLForecast(
    model = [],
    freq = 1,
    lags = [1, 24],
    target_transforms=[Differences([24])],
)
prep = fcst.preprocess(df)
print(prep)
# lags features 与预测变量的相关性
print(prep.drop(columns=["unique_id", "ds"]).corr()["y"])


# ------------------------------
# model training
# ------------------------------



# ------------------------------
# model forecasting
# ------------------------------



# ------------------------------
# model save and load
# ------------------------------



# ------------------------------
# update series's value
# ------------------------------



# ------------------------------
# model evaluation
# ------------------------------


# 测试代码 main 函数
def main():
    df = load_data()
    print(df)

if __name__ == "__main__":
    main()
