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

import pandas as pd
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt


def timeseries_line_plot():
    pass


def timeseries_hist_plot():
    pass




if __name__ == "__main__":
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
    # ------------------
    # temperature_data.plot()
    # temperature_data.plot(style = "k.")
    # temperature_data.plot(style = "k-")
    # plt.show()
    # ------------------
    # line group by year
    # ------------------
    # groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    # years = pd.DataFrame()
    # for name, group in groups:
    #     years[name.year] = group.values
    # print(years)
    # years.plot(subplots = True, legend = True)
    # plt.show()
    # -----------------------------------------------
    # Hist
    # -----------------------------------------------
    # hist
    # ------------------
    # temperature_data.hist()
    # ------------------
    # kde
    # ------------------
    # temperature_data.plot(kind = "kde")
    # plt.show()
    # -----------------------------------------------
    # Boxplot
    # -----------------------------------------------
    # boxplot
    # ------------------
    # temperature_data = pd.DataFrame(temperature_data)
    # temperature_data.boxplot()
    # plt.show()
    # ------------------
    # boxplot group by year
    # ------------------
    # groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    # years = pd.DataFrame()
    # for name, group in groups:
    #     years[name.year] = group.values
    # years.boxplot()
    # plt.show()
    # ------------------
    # boxplot group by month
    # ------------------
    # temperature_data_1990 = temperature_data["1990"]
    # groups = temperature_data_1990.groupby(pd.Grouper(freq = "M"))
    # months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis = 1)
    # months = pd.DataFrame(months)
    # months.columns = range(1, 13)
    # months.boxplot()
    # plt.show()
    # -----------------------------------------------
    # Heat map
    # -----------------------------------------------
    # heat map group by year
    # ------------------
    # groups = temperature_data.groupby(pd.Grouper(freq = "A"))
    # years = pd.DataFrame()
    # for name, group in groups:
    #     years[name.year] = group.values
    # years = years.T
    # plt.matshow(years, interpolation = None, aspect = "auto")
    # plt.show()
    # ------------------
    # heat map group by month
    # ------------------
    # temperature_data_1990 = temperature_data["1990"]
    # groups = temperature_data_1990.groupby(pd.Grouper(freq = "M"))
    # months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis = 1)
    # months = pd.DataFrame(months)
    # months.columns = range(1, 13)
    # plt.matshow(months, interpolation = None, aspect = "auto")
    # plt.show()
    # -----------------------------------------------
    # Lagged scatter plot 滞后散点图
    # -----------------------------------------------
    # lagged scatter plot
    # ------------------
    # lag_plot(temperature_data)
    # plt.show()
    # ------------------
    # lagged plot
    # ------------------
    # values = pd.DataFrame(temperature_data.values)
    # lags = 7
    # columns = [values]
    # for i in range(1,(lags + 1)):
    #     columns.append(values.shift(i))
    # dataframe = pd.concat(columns, axis=1)
    # columns = ['t+1']
    # for i in range(1,(lags + 1)):
    #     columns.append('t-' + str(i))
    # dataframe.columns = columns
    # plt.figure(1)
    # for i in range(1,(lags + 1)):
    #     ax = plt.subplot(240 + i)
    #     ax.set_title('t+1 vs t-' + str(i))
    #     plt.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)
    # plt.show()
    # -----------------------------------------------
    # autocorrelation plot 自相关图
    # -----------------------------------------------
    autocorrelation_plot(temperature_data)
    plt.show()
