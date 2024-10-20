# -*- coding: utf-8 -*-

# ***************************************************
# * File        : mlforecast_fit.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091101
# * Description : mlforecast quick start(local)
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

import pandas as pd
import matplotlib.pyplot as plt
from utilsforecast.plotting import plot_series

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
df = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv", parse_dates=["ds"])
print(df.head())
print(df.shape)
print(df["unique_id"].value_counts())

# data visual
fig = plot_series(df)

# model
fcst = MLForecast(
    models = LinearRegression(),
    freq = "MS",  # series has a monthly frequency
    lags = [12],
    target_transforms = [Differences([1])],  # remove trend
)

# model train
fcst.fit(df)

# model predict
preds = fcst.predict(h = 12)
print(preds)

# result visual
fig = plot_series(df, preds)
plt.show();



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
