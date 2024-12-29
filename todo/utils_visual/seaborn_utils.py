import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegird")


def scatter_matrix(df, cagegorical = None):
    """
    # 散点图矩阵
    """
    if cagegorical:
        sns.pairplot(df, hue = cagegorical, size = 2.5)
    else:
        sns.pairplot(df, size = 2.5)


def histogram_plot(df, row, col, categorical, imgpath = None):
    """
    分面直方图
    """
    grid = sns.FacetGrid(df, row, col, margin_titles = True)
    grid.map(plt.hist, categorical, bins = np.linspace(0, 40, 15))


def timeseries_plot(df, ts, y, title):
    fig, ax = plt.subplots(figsize = (20, 7))
    chart = sns.lineplot(x = ts, y = y, data = df)
    chart.set_title("%s Timeseries Data" % y, fontsize = 15)
    plt.show();


def bar_plot(df, df_new, x, y, categorical):
    fig, ax = plt.subplots(figsize = (20, 5))
    palette = sns.color_palette("mako_r", 4)
    a = sns.barplot(x = "month", y = "Sales", hue = "year", data = df_new)
    a.set_title("Store %s Data" % y, fontsize = 15)
    plt.legend(loc = "upper right")
    plt.show()


def bar_plots(df, x, y, xlabel, ylabel, nrows, title):
    fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=nrows)
    fig.set_size_inches(20,30)
    for ax in [ax1,ax2,ax3,ax4]:
        sns.barplot(data = df, x = x,y = y, ax = ax)
        ax.set(xlabel = xlabel, ylabel = ylabel)
        ax.set_title(title, fontsize=15)


def resid_plot(df, x, y):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.residplot(x = x, y = y, data = df)


def reg_plot(df, x, y):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(
        x = x, y = y, data = df, order = 0, 
        scatter_kws = {
            "s": 20
        }
    )

