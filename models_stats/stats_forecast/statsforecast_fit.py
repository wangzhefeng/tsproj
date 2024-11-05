# -*- coding: utf-8 -*-

# ***************************************************
# * File        : statsforecast_fit.py
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

import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoRegressive, AutoCES, AutoETS, AutoMFLES, AutoTBATS, AutoTheta

os.environ["NIXTLA_ID_AS_COL"] = "1"
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
df = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv", parse_dates = ["ds"])
print(df.head())
print(df.shape)
print(df["unique_id"].value_counts())

# model training
sf = StatsForecast(
    models = [
        AutoARIMA(season_length=12),
    ],
    freq = "M",
)
sf.fit(df)

# model predict
forecast_df = sf.predict(h = 12, level = [90])
print(forecast_df)

# model predict result visual
sf.plot(df, forecast_df, level = [90])
plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
