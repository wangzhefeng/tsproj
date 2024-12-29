# -*- coding: utf-8 -*-


# ***************************************************
# * File        : temp.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-10-05
# * Version     : 0.1.100519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data read
series = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    header = 0,
    index_col = 0,
    parse_dates = True,
    date_parser = lambda dates: pd.to_datetime(dates, format = '%Y-%m-%d'),
    squeeze = True,
)
# data view
print(series.head())

df = pd.DataFrame()
df["Date"] = series.index
df["month"] = [series.index[i].month for i in range(len(series))]
df["day"] = [series.index[i].day for i in range(len(series))]
df["temperature"] = [series[i] for i in range(len(series))]
print(df.head())

# ------------------------------
# data plot
# ------------------------------
series.plot()
plt.show()

groups = series.groupby(pd.Grouper(freq = "A"))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.plot(subplots = True, legend = False)
plt.show()

series.plot(kind = "hist")
plt.show()











 
# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

