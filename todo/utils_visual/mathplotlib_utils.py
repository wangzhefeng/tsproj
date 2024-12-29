import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

# config
plt.style.use("classic")
plt.style.use("seaborn-whitegrid")


def math_function_plot(func, x, imgpath = None):
    """
    matlab plot style
    """
    fig, axs = plt.subplots(figsize = (10, 7), sharey = True, tight_layout = True)
    plt.plot(x, func(x), label = "x")
    
    plt.title("function of x")
    plt.xlabel("x")
    plt.ylabel("function(x)")
    plt.legend()
    if imgpath:
        plt.savefig(imgpath)
    plt.show();


def histogram_plot(df, *xlabels, xlim = None, imgpath = None):
    """
    pandas plot style
    """
    fig, axs = plt.subplots(figsize = (10, 7), sharey = True, tight_layout = True)
    for xlabel in xlabels:
        df[xlabel].plot(kind = "hist", bins = 20, color = "steelblue", edgecolor = "black", density = True, label = "Histogram")
        df[xlabel].plot(kind = "kde", color = "red", label = "KDE")
        # kwargs = dict(histtype = "", alpha = 0.3, normed = True, bins = 40)
        # plt.hist(data, **kwargs)
        # counts, bin_edges = np.histogram(data, bins=5)
        plt.xlabel("%s" % xlabel)
        plt.ylabel("Count")
        plt.title("%s Histogram" % xlabel)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    plt.legend()
    if imgpath:
        plt.savefig(imgpath)
    plt.show();


def timeseries_plot(df, xlabel, *ylabels, imgpath = None):
    """
    matlab plot style
    """
    fig, axs = plt.subplots(figsize = (30, 7), sharey = True, tight_layout = True)
    for ylabel in ylabels:
        plt.plot_date(df[xlabel], df[ylabel], linestyle = "solid", label = ylabel)
    
    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter("%Y-%B-%d")
    plt.gca().xaxis.set_major_formatter(date_format)
    
    plt.title(ylabel)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    if imgpath:
        fig.save(imgpath)
    plt.show();


def scatter_plot(df, xlabel, *ylabels, imgpath = None):
    """
    matlab plot style
    """
    fig, axs = plt.subplots(figsize = (10, 7), sharey = True, tight_layout = True)
    for ylabel in ylabels:
        plt.scatter(df[xlabel], df[ylabel], marker = "o", label = ylabel)
        # plt.plot(df[xlabel], df[ylabel], "", color, label = "marker='{}'".format(marker))
        # axs.plot(df[xlabel], df[ylabel], "", color, label = ylabel)
        
        plt.title("%s VS %s" % (xlabel, ylabel))
        plt.xlabel("%s" % xlabel)
        # plt.ylabel("%s" % ylabel)
        plt.legend()
        if imgpath:
            plt.savefig(imgpath)
        plt.show()


def scatter_many_plot(df, x, y, labels):
    if len(labels["x"]) == len(labels["y"]):
        num_plots = len(labels["x"])
        fig, axs = plt.subplots(figsize = (8 * num_plots, 8))
        for i in range(num_plots):
            plt.subplot(1, num_plots, i)
            plt.scatter(df[x[i]], df[y[i]])
            plt.title("%s VS %s" % (x[i], y[i]))
            plt.xlabel(x[i])
            plt.ylabel(y[i])
        plt.show()



def bar_plot():
    pass




