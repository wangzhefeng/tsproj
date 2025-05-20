# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_visual.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090823
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
from typing import List, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import seaborn as sns

# 绘图风格
# plt.style.use("ggplot")  # style sheet config
# plt.style.use("classic")  # style sheet config
# sns.set_style("darkgrid")  # sns style
# 字体尺寸设置
# plt.rcParams["font.size"] = 10  # mpl font size
# sns.mpl.rc("font", size = 14)  # sns font size
title_fontsize = 13
label_fontsize = 7
# figure 设置
# sns.mpl.rc("figure", figsize = (16, 6))
plt.tight_layout()
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True
# 字体设置
plt.rcParams['font.sans-serif']=['SimHei', "Arial Unicode MS"]  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来显示负号
plt.rcParams["font.family"].append("SimHei")  # 处理 matplotlib 字体问题

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_array_curve(data_list: List, ycol: str, title: str):
    """
    绘制拱顶温度、烟道温度曲线
    """
    # data
    df = pd.DataFrame({ycol: data_list})
    # plot
    fig = plt.figure()
    plt.plot(df.index, df[ycol], marker = ".", linestyle = "-.")
    plt.title(label = title)
    plt.show();


def plot_df_curve(df: pd.DataFrame, ycol: str, title: str):
    """
    绘制曲线
    """
    # plot
    fig = plt.figure()
    plt.plot(df.index, df[ycol], marker = ".", linestyle = "-.")
    plt.title(label = title)
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
