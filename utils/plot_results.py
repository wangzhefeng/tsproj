# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052220
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

"""
时间序列数据可视化

1.line plot
2.lag plot
3.autocorrelation plot
4.histograms plot
5.density plot
6.box plot
7.whisker plot
8.heat map plot
"""

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error

# 绘图风格
plt.style.use("seaborn-v0_8-whitegrid")  # "ggplot", "classic", "darkgrid"

# 用来正常显示中文标签
plt.rcParams["font.sans-serif"]=["SimHei"]  # 'Arial Unicode MS'
# 处理 matplotlib 字体问题
plt.rcParams["font.family"].append("SimHei")
# 用来显示负号
plt.rcParams["axes.unicode_minus"] = False

# TODO 字体尺寸设置
# plt.rcParams["font.size"] = 10  # mpl font size
# sns.mpl.rc("font", size = 14)  # sns font size
title_fontsize = 13
label_fontsize = 7

# figure 设置
# sns.mpl.rc("figure", figsize = (16, 6))
# plt.tight_layout()
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams["axes.grid"] = True
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4.5),
    titleweight="bold",
    titlesize=18,
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
    grid=True,
)


train_scatter_plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Train trues",
)
test_scatter_plot_params = dict(
    color="C2",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Test trues",
)
fit_line_plot_params = dict(
    color="C0",
    linewidth=2,
    legend=True,
    label="Train preds",
)
pred_line_plot_params = dict(
    color="C1",
    linewidth=2,
    legend=True,
    label="Test preds",
)
fore_line_plot_params = dict(
    color="C3",
    linewidth=2,
    legend=True,
    label="Forecast",
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def predict_result_visual(preds: np.array, trues: np.array, path='./path/test.pdf'):
    """
    Results visualization
    """
    fig = plt.figure(figsize=(25, 8))
    plt.plot(trues, lw=1.2, label='Trues')
    plt.plot(preds, lw=1.2, label='Preds', ls="-.")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title('Trues and Preds Timeseries Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.show();


def plot_result_with_interval(X_train, y_train, y_test, y_pred, model):
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.values, label="test actual", lw=2.0)
    plt.plot(y_pred,        label="test pred",   lw=2.0, ls="-.")
    
    # 模型预测区间
    plot_intervals = False
    if plot_intervals:
        # data split
        tscv = TimeSeriesSplit(n_splits=5)
        # tscv.split(series)
        cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        deviation = np.sqrt(cv.std())
        scale = 1.96
        lower = y_pred - (scale * deviation)
        upper = y_pred + (scale * deviation)
        plt.plot(lower, linestyle=":", alpha=0.6, color="C2", label="lower bond")
        plt.plot(upper, linestyle=":", alpha=0.6, color="C2", label="upper bond")
    
    # 异常值
    plot_anomalies = False
    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(y_test))
        anomalies[y_test < lower] = y_test[y_test < lower]
        anomalies[y_test > upper] = y_test[y_test > upper]
        plt.plot(anomalies, "o", markersize=3, color="C3", label="Anomalies")
    
    plt.title(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test.values, y_pred):.2f}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)
    plt.show();


def rolling_windows_predict_result_visual(df: pd.DataFrame, path="./path/test.pdf"):
    """
    测试集滑动窗口预测结果可视化

    Args:
        df (pd.DataFrame): _description_
        path (str, optional): _description_. Defaults to "./path/test.pdf".
    """
    # 数据处理
    df["ds"] = pd.to_datetime(df["ds"])
    df["Y_trues"] = df["Y_trues"].apply(lambda x: round(x, 2))
    df["Y_preds"] = df["Y_preds"].apply(lambda x: round(x, 2))
    # 画图
    fig = plt.figure(figsize=(25, 8))
    plt.plot(df["ds"], df["Y_trues"], linewidth="1.2", label="tures")
    plt.plot(df["ds"], df["Y_preds"], linewidth="1.2", label="preds")
    plt.legend()
    plt.xlabel("Time", fontdict={"size": 15})
    plt.ylabel("Value", fontdict={"size": 15})
    plt.title("Model Forecast Timeseries Plot", fontdict={"size": 20})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.show();


def series_plot(df: pd.DataFrame, time_col, value_col):
    """
    单时序图

    Args:
        df (_type_): 时序数据
        time_col (_type_): 时间变量
        value_col (_type_): 待预测变量
    """
    fig = plt.figure(figsize=(15, 5))
    plt.plot(df[time_col], df[value_col], marker = ".", linestyle = "-.")
    plt.legend()
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.title(f"{value_col} 时序图")
    plt.tight_layout()
    plt.grid()
    plt.show();


def scatter_reg_plot(data, x, y, aspect: bool=False):
    """
    散点图回归拟合曲线
    """
    fig, ax = plt.subplots()
    # ax.plot(x, y, data=data, color="0.75")
    ax = sns.regplot(x=x, y=y, data=data, ci=None, scatter_kws=dict(color="0.25", s=9))
    if aspect:
        ax.set_aspect("equal")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x.capitalize()} Plot of {y.capitalize()}")
    plt.show();


def model_result_plot(y_train: pd.Series, y_test: pd.Series, 
                      y_fit: pd.Series, y_pred: pd.Series, y_fore: pd.Series, 
                      xlabel: str, ylabel: str, title: str):
    """
    时间序列模型结果可视化

    Args:
        y_train (pd.Series): _description_
        y_test (pd.Series): _description_
        y_fit (pd.Series): _description_
        y_pred (pd.Series): _description_
        y_fore (pd.Series): _description_
        xlabel (str): _description_
        ylabel (str): _description_
        title (str): _description_
    """
    import matplotlib as mpl
    font_name = [
        "Arial Unicode MS", 
        # "SimHei"
    ]
    mpl.rcParams["font.sans-serif"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False
    # result plot
    fig, ax = plt.subplots()
    if y_train is not None:
        ax = y_train.plot(ax=ax, **train_scatter_plot_params)
    if y_test is not None:
        ax = y_test.plot(ax=ax, **test_scatter_plot_params)
    if y_fit is not None:
        ax = y_fit.plot(ax=ax, **fit_line_plot_params)
    if y_pred is not None:
        ax = y_pred.plot(ax=ax, **pred_line_plot_params)
    if y_fore is not None:
        ax = y_fore.plot(ax=ax, **fore_line_plot_params)
    _ = ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show();


# TODO
def plot_results(y_preds, y_trues, title):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(y_trues, label='True Data')
    plt.plot(y_preds, label='Prediction')
    plt.legend()
    plt.title(title)
    plt.savefig(f'images/{title}_results.png')
    plt.show();


# TODO
def plot_results_multiple(preds, trues, preds_len, title):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(trues, label = 'True Data')
    for i, data in enumerate(preds):
        padding = [None for p in range(i * preds_len)]
        plt.plot(padding + data, label = 'Prediction')
    plt.legend()
    plt.title(title)
    plt.savefig(f'images/{title}_results_multiple.png')
    plt.show();


def plot_heatmap(dfs: List, 
                 stat: str = "corr",  # 协方差矩阵: "cov"
                 method: str = "pearson", 
                 figsize: Tuple = (5, 5), 
                 titles: List[str] = [], 
                 img_file_name: str = ""):
    """
    相关系数、协方差矩阵热力图
    """
    fig, axes = plt.subplots(nrows = 1, ncols = len(dfs), figsize = figsize)
    stat_matrixs = []
    for idx, df, title in zip(range(len(dfs)), dfs, titles):
        # ax
        ax = axes[idx] if len(dfs) > 1 else axes
        # 计算相关系数矩阵或协方差矩阵
        if stat == "corr":
            stat_matrix = df.corr(method)#.sort_values(by = sort_col_list, ascending = False)
        elif stat == "cov":
            stat_matrix = df.cov()#.sort_values(by = sort_col_list, ascending = False)
        # 绘制相关系数矩阵热力图
        sns.heatmap(
            data = stat_matrix, annot = True, annot_kws = {"size": 8}, 
            square = True, cmap = sns.diverging_palette(20, 220, n = 256), 
            linecolor = 'w', center = 0, vmin = -1, vmax = 1, 
            fmt = ".2f", cbar = False, ax = ax,
        )
        ax.xaxis.tick_top()
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f"{title}相关系数矩阵热力图", fontsize = title_fontsize)
        # 收集相关系数、协方差矩阵
        stat_matrixs.append(stat_matrix)
    # 图像显示
    plt.show()
    # 图像保存
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)
        
    return stat_matrixs


def plot_scatter(data, x: str, y: str, 
                 logx = False, logy = False,
                 xtick_major = None, xtick_minor = None,
                 ytick_major = None, ytick_minor = None,
                 hline_ll = None, hline_ul = None,
                 vline_ll = None, vline_ul = None,
                 figsize = (8, 8),
                 title = None):
    fig, ax = plt.subplots(figsize = figsize)
    # xtick
    if xtick_major:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_major))
    if xtick_minor:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtick_minor))
    # ytick
    if ytick_major:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_major))
    if ytick_minor:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_minor))
    # grid
    ax.grid(True, which = "both", ls = "dashed")
    # plot
    data.plot(
        kind = "scatter",
        x = x,
        y = y,
        s = 20,
        alpha = 0.7,
        edgecolors = "white",
        legend = True,
        logx = logx,
        logy = logy,
        ax = ax,
    )
    # xlabel
    plt.setp(ax.get_xmajorticklabels(), rotation = 90.0)
    plt.setp(ax.get_xminorticklabels(), rotation = 0.0)
    # hline and vline
    if hline_ll:
        plt.axhline(y = hline_ll, color = "red", linestyle = "--")
    if hline_ul:
        plt.axhline(y = hline_ul, color = "red", linestyle = "--")
    if vline_ll:
        plt.axvline(x = vline_ll, color = "darkgreen", linestyle = "--")
    if vline_ul:
        plt.axvline(x = vline_ul, color = "darkgreen", linestyle = "--")
    # title
    plt.title(title)
    plt.show();


def plot_scatter_multicols(df: pd.DataFrame, 
                           xcols: List[str], 
                           ycols: List[str], 
                           cate_cols: List[str], 
                           figsize: Tuple = (5, 5), 
                           img_file_name: Union[str, any] = None):
    """
    散点图
    scatter legend link ref:
    https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    """
    fig, axes = plt.subplots(nrows = 1, ncols = len(xcols), figsize = figsize)
    for idx, xcol, ycol, cate_col in zip(range(len(xcols)), xcols, ycols, cate_cols):
        # ax
        ax = axes[idx] if len(xcols) > 1 else axes
        # 散点图
        if cate_col is not None:
            sns.scatterplot(data = df, x = xcol, y = ycol, hue = cate_col, ax = ax)
        else:
            sns.scatterplot(data = df, x = xcol, y = ycol, ax = ax)
        # label
        ax.set_xlabel(xcol, fontsize = label_fontsize)
        ax.set_ylabel(ycol, fontsize = label_fontsize)
        # title
        ax.set_title(f"{xcol} 与 {ycol} 相关关系散点图", fontsize = title_fontsize)
        # legend
        ax.legend(loc = "best")
    # 图像显示
    plt.show()
    # 图像保存
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_scatter_reg(df: pd.DataFrame, 
                     xcols: List[str], 
                     ycols: List[str],
                     figsize: Tuple = (5, 5),
                     xtick_major = None, xtick_minor = None,
                     ytick_major = None, ytick_minor = None,
                     hline_ll: int = None, hline_ul: int = None,
                     vline_ll: int = None, vline_ul: int = None,
                     title: str = "",
                     img_file_name: str = None):
    """
    带拟合曲线的散点图
    """
    fig, axes = plt.subplots(nrows = 1, ncols = len(xcols), figsize = figsize)
    for idx, xcol, ycol in zip(range(len(xcols)), xcols, ycols):
        # ax
        ax = axes[idx] if len(xcols) > 1 else axes
        # xtick
        if xtick_major:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_major))
        if xtick_minor:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtick_minor))
        # ytick
        if ytick_major:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_major))
        if ytick_minor:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_minor))
        # 带拟合曲线的散点图
        sns.regplot(
            data = df, 
            x = xcol,
            y = ycol,
            # robust = True,
            lowess = True, 
            line_kws = {"color": "C2"},  
            ax = ax
        )
        # hline and vline
        if hline_ll:
            plt.axhline(y = hline_ll, color = "red", linestyle = "--")
        if hline_ul:
            plt.axhline(y = hline_ul, color = "red", linestyle = "--")
        if vline_ll:
            plt.axvline(x = vline_ll, color = "darkgreen", linestyle = "--")
        if vline_ul:
            plt.axvline(x = vline_ul, color = "darkgreen", linestyle = "--")
        # xlabel adjust
        plt.setp(ax.get_xmajorticklabels(), rotation = 90.0)
        plt.setp(ax.get_xminorticklabels(), rotation = 0.0)
    # title
    plt.title(f"{title}相关关系图")
    # show
    plt.show()
    # save
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_scatter_lm(df: pd.DataFrame, 
                    xcol: str, 
                    ycol: str, 
                    cate_col: str, 
                    figsize: Tuple = (5, 5),
                    img_file_name: str = None):
    """
    带拟合曲线的散点图
    """
    fig, axes = plt.figure(nrows = 1, ncols = 1, figsize = figsize)
    # plot
    if cate_col is not None:
        sns.lmplot(data = df, x = xcol, y = ycol, hue = cate_col, robust = True, ax = axes)
    else:
        sns.lmplot(data = df, x = xcol, y = ycol, robust = True, ax = axes)
    # title
    plt.title(f"{xcol} 与 {ycol} 相关关系散点图", fontsize = title_fontsize)
    # 图像显示
    plt.show()
    # 图像保存
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_scatter_matrix(df: pd.DataFrame,
                        cols: List[str],
                        figsize: Tuple = (10, 10),
                        xlabel: str = None,
                        ylabel: str = None,
                        title: str = "",
                        img_file_name: str = None):
    """
    散点图矩阵
    """
    # figure
    fig, axes = plt.subplots(figsize = figsize)
    # plot
    sns.pairplot(data = df[cols], kind = "reg", diag_kind = "kde", corner = True)
    # label
    axes.set_xlabel(xlabel, fontsize = label_fontsize)
    axes.set_ylabel(ylabel, fontsize = label_fontsize)
    # title
    axes.set_title(f"{title} 的散点图矩阵", fontsize = title_fontsize)
    # show
    plt.show()
    # save
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_timeseries(data: List = None,
                    ts_cols: List = None,
                    xtick_major = None, xtick_minor = None, 
                    ytick_major = None, ytick_minor = None, 
                    hline_ll = None, hline_ul = None,
                    vline_ll = None, vline_ul = None,
                    figsize = (28, 10),
                    title = None):
    """
    时间序列图
    """
    fig, ax = plt.subplots(figsize = figsize)
    # xtick
    if xtick_major:
        ax.xaxis.set_major_locator(dates.DayLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    if xtick_minor:
        ax.xaxis.set_minor_locator(dates.HourLocator())
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%Y-%m-%d %H"))
    # ytick
    if ytick_major:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_major))
    if ytick_minor:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_minor))
    # grid
    ax.grid(True, which = "both", ls = "dashed")
    # plot
    # 1图多曲线(一个数据源)
    if len(data) == 1:
        data[0][ts_cols].plot(legend = True, ax = ax)
    # 1图1曲线(两个数据源)
    if len(data) == 2 and len(ts_cols) == 1:
        data[0][ts_cols].plot(legend = True, ax = ax)
        data[1][ts_cols].plot(legend = True, ax = ax)
    # 1图2曲线(两个数据源)
    if len(data) == 2 and len(ts_cols) == 2:
        data[0][[ts_cols[0]]].plot(legend = True, ax = ax)
        data[1][[ts_cols[1]]].plot(legend = True, ax = ax)
    # xlabel
    plt.setp(ax.get_xmajorticklabels(), rotation = 0.0)
    plt.setp(ax.get_xminorticklabels(), rotation = 0.0)
    # hline and vline
    if hline_ll:
        plt.axhline(y = hline_ll, color = "red", linestyle = "--")
    if hline_ul:
        plt.axhline(y = hline_ul, color = "red", linestyle = "--")
    if vline_ll:
        plt.axvline(x = vline_ll, color = "darkgreen", linestyle = "--")
    if vline_ul:
        plt.axvline(x = vline_ul, color = "darkgreen", linestyle = "--")
    # title
    plt.title(f"{title}时序图")
    # show
    plt.show();


def plot_timeseries_multicols(df: pd.DataFrame, 
                              n_rows_cols: List[int],
                              ycols: List[str], 
                              cate_col: str = None, 
                              figsize: Tuple = (7, 5), 
                              img_file_name: str = None):
    """
    时间序列图
    """
    fig, axes = plt.subplots(nrows = n_rows_cols[0], ncols = n_rows_cols[1], figsize = figsize)
    for idx, ycol in enumerate(ycols):
        # ax
        ax = axes[idx] if len(ycols) > 1 else axes
        # 线形图
        if cate_col is not None:
            sns.lineplot(data = df, x = df.index, y = ycol, hue = cate_col, marker = ",", ax = ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
            ax.set_title(f"{ycol} 在不同 {cate_col} 下的对比图", fontsize = title_fontsize)
        else:
            sns.lineplot(data = df, x = df.index, y = ycol, marker = ",", ax = ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
            ax.set_title(f"{ycol} 的时间序列图", fontsize = title_fontsize)
    # show
    plt.show()
    # save
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_distributed(df: pd.DataFrame, 
                     xcols: List[str], 
                     cate_cols: List[str], 
                     figsize: Tuple = (5, 5), 
                     img_file_name: str = None):
    """
    分布图（直方图、KDE 曲线）

    Args:
        df (pd.DataFrame): _description_
        xcols (List[str]): _description_
        cate_cols (List[str]): _description_
        figsize (Tuple, optional): _description_. Defaults to (5, 5).
        img_file_name (str, optional): _description_. Defaults to None.
    """
    # figure
    fig, axes = plt.subplots(nrows = 1, ncols = len(xcols), figsize = figsize)
    for idx, xcol, cate_col in zip(range(len(xcols)), xcols, cate_cols):
        # ax
        ax = axes[idx] if len(xcols) > 1 else axes
        # hist plot
        if cate_col is not None:
            sns.histplot(data = df, x = xcol, hue = cate_col, kde = True, ax = ax)
        else:
            sns.histplot(data = df, x = xcol, kde = True, ax = ax)
        # label
        ax.set_xlabel(xcol, fontsize = label_fontsize)
        # title
        ax.set_title(f"{xcol} 分布直方图", fontsize = title_fontsize)
    # show
    plt.show()
    # save
    if img_file_name is not None:
        fig.get_figure().savefig(f'imgs/{img_file_name}.png', bbox_inches = 'tight', transparent = True)


def plot_scatter(data, x: str, y: str, cate_col: str = None,
                 logx = False, logy = False,
                 xtick_major = None, xtick_minor = None,
                 ytick_major = None, ytick_minor = None,
                 hline_ll = None, hline_ul = None,
                 vline_ll = None, vline_ul = None,
                 figsize = (8, 8),
                 title = None):
    fig, ax = plt.subplots(figsize = figsize)
    # xtick
    if xtick_major:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_major))
    if xtick_minor:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtick_minor))
    # ytick
    if ytick_major:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_major))
    if ytick_minor:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ytick_minor))
    # grid
    ax.grid(True, which = "both", ls = "dashed")
    # plot
    if cate_col is None:
        data.plot(
            kind = "scatter",
            x = x,
            y = y,
            s = 20,
            alpha = 0.5,
            edgecolors = "white",
            legend = True,
            logx = logx,
            logy = logy,
            ax = ax,
        )
    else:
        sns.scatterplot(data=data, x=x, y=y, hue=cate_col, ax = ax)
    # xlabel
    plt.setp(ax.get_xmajorticklabels(), rotation = 90.0)
    plt.setp(ax.get_xminorticklabels(), rotation = 0.0)
    # hline and vline
    if hline_ll:
        plt.axhline(y = hline_ll, color = "red", linestyle = "--")
    if hline_ul:
        plt.axhline(y = hline_ul, color = "red", linestyle = "--")
    if vline_ll:
        plt.axvline(x = vline_ll, color = "darkgreen", linestyle = "--")
    if vline_ul:
        plt.axvline(x = vline_ul, color = "darkgreen", linestyle = "--")
    # title
    plt.title(title)
    plt.show();




# 测试代码 main 函数
def main():
    import pandas as pd
    from pandas.plotting import lag_plot
    from pandas.plotting import autocorrelation_plot
    # pandas plot
    pd.plotting.register_matplotlib_converters()
    
    # data
    temperature_data = pd.read_csv(
        filepath_or_buffer = "/Users/zfwang/machinelearning/datasets/data_visualization/daily-minimum-temperatures.csv", 
        header = 0, 
        index_col = 0, 
        parse_dates = True, 
        squeeze = True
    )
    # -----------------------------------------------
    # Line
    # -----------------------------------------------
    # line
    temperature_data.plot()
    temperature_data.plot(style = "k.")
    temperature_data.plot(style = "k-")
    plt.show()
    # line group by year
    groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    years = pd.DataFrame()
    for name, group in groups:
        years[name.year] = group.values
    print(years)
    years.plot(subplots = True, legend = True)
    plt.show()
    # -----------------------------------------------
    # Hist
    # -----------------------------------------------
    # hist
    temperature_data.hist()
    # kde
    temperature_data.plot(kind = "kde")
    plt.show()
    # -----------------------------------------------
    # Boxplot
    # -----------------------------------------------
    # boxplot
    temperature_data = pd.DataFrame(temperature_data)
    temperature_data.boxplot()
    plt.show()
    # boxplot group by year
    groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    years = pd.DataFrame()
    for name, group in groups:
        years[name.year] = group.values
    years.boxplot()
    plt.show()
    # ------------------
    # boxplot group by month
    # ------------------
    temperature_data_1990 = temperature_data["1990"]
    groups = temperature_data_1990.groupby(pd.Grouper(freq = "M"))
    months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis = 1)
    months = pd.DataFrame(months)
    months.columns = range(1, 13)
    months.boxplot()
    plt.show()
    # -----------------------------------------------
    # Heat map
    # -----------------------------------------------
    # heat map group by year
    groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    years = pd.DataFrame()
    for name, group in groups:
        years[name.year] = group.values
    years = years.T
    plt.matshow(years, interpolation = None, aspect = "auto")
    plt.show()
    # heat map group by month
    temperature_data_1990 = temperature_data["1990"]
    groups = temperature_data_1990.groupby(pd.Grouper(freq = "M"))
    months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis = 1)
    months = pd.DataFrame(months)
    months.columns = range(1, 13)
    plt.matshow(months, interpolation = None, aspect = "auto")
    plt.show()
    # -----------------------------------------------
    # Lagged scatter plot 滞后散点图
    # -----------------------------------------------
    # lagged scatter plot
    lag_plot(temperature_data)
    plt.show()
    # lagged plot
    values = pd.DataFrame(temperature_data.values)
    lags = 7
    columns = [values]
    for i in range(1,(lags + 1)):
        columns.append(values.shift(i))
    dataframe = pd.concat(columns, axis=1)
    columns = ['t+1']
    for i in range(1,(lags + 1)):
        columns.append('t-' + str(i))
    dataframe.columns = columns
    plt.figure(1)
    for i in range(1,(lags + 1)):
        ax = plt.subplot(240 + i)
        ax.set_title('t+1 vs t-' + str(i))
        plt.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)
    plt.show()
    # -----------------------------------------------
    # autocorrelation plot 自相关图
    # -----------------------------------------------
    autocorrelation_plot(temperature_data)
    plt.show()

if __name__ == "__main__":
    main()
