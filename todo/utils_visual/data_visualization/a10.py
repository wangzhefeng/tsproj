# -*- coding: utf-8 -*-


# ***************************************************
# * File        : a10.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-03
# * Version     : 0.1.120317
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# 时间解析
from dateutil.parser import parse

# 数据可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 数据处理
import numpy as np
import pandas as pd
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 120,
})


# ------------------------------
# 1.读取数据
# ------------------------------
# 1.1 时间序列数据
df_a10 = pd.read_csv(
    # "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv", 
    "/Users/zfwang/machinelearning/datasets/a10/a10.csv",
    parse_dates=['date'],
    # index_col = "date",
)
df_a10.head()


# 1.2 面板数据
df_market_arrivals = pd.read_csv(
    # "https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv"
    "/Users/zfwang/machinelearning/datasets/a10/MarketArrivals.csv"
)
df_market_arrivals = df_market_arrivals.loc[df_market_arrivals.market == "MUMBAI", :]
df_market_arrivals.head()


# 2.时间序列可视化
def plot_df(date, value, title = "", xlabel = "Date", ylabel = "Value", dpi = 100):
    plt.figure(figsize = (16, 5), dpi = dpi)
    plt.plot(date, value, color = "tab:red")
    plt.gca().set(title = title, xlabel = xlabel, ylabel = ylabel)
    plt.show()

plot_df(
    x = df_a10["date"].values, 
    y = df_a10["value"].values, 
    title = "Monthly anti-diabetic drug sales in Australia from 10992 to 2008"
)


def plot_two_side_df(date, value, title = "", xlabel = "Date", ylabel = "Value", dpi = 100):
    fig, ax = plt.subplots(1, 1, figsize = (16, 5), dpi = dpi)
    plt.fill_between(x = date, y1 = value, y2 = -value, linewidth = 2, color = "seagreen")
    plt.ylim()
    plt.title(title, fontsize = 16)
    plt.hlines(y = 0, xmin = np.min(x), xmax = np.max(x), linewidth = 0.5)
    plt.show()
    
plot_two_side_df(
    x = df_a10["date"].values, 
    y = df_a10["value"].values,
    title = "Drug Sale(Two Side View)"
)


df_a10["year"] = [d.year for d in df_a10.date]
df_a10["month"] = [d.strftime("%b") for d in df_a10.date]
years = df_a10["year"].unique()

plt.figure(figsize = (16, 12), dpi = 100)
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)
for i, y in enumerate(years):
    if i > 0:
        plt.plot(
            "month", 
            "value", 
            data = df_a10.loc[df_a10.year == y, :], 
            color = mycolors[i], 
            label = y
        )
        plt.text(
            df_a10.loc[df_a10.year == y, :].shape[0] - 0.9, 
            df_a10.loc[df_a10.year == y, "value"][-1:].values[0], 
            y, 
            fontsize = 12, 
            color = mycolors[i]
        )

plt.gca().set(xlim = (-0.3, 11), ylim = (2, 30), xlabel = "$Month$", ylabel = "$Drug Sales$")
plt.yticks(fontsize = 12)
plt.title("Seasonal Plot of Drug Sales Time Series", fontsize = 20)


fig, axes = plt.subplots(1, 2, figsize = (20, 7), dpi = 80)
sns.boxplot(x = "year", y = "value", data = df_a10, ax = axes[0])
sns.boxplot(x = "month", y = "value", data = df_a10.loc[~df_a10.year.isin([1991, 2008]), :], ax = axes[1])
axes[0].set_title("Year-wise Box Plot\n(The Trend)", fontsize = 18)
axes[1].set_title("Month-wise Box Plot\n(The Seasonality)", fontsize = 18)
plt.show()

