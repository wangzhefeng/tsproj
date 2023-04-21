# -*- coding: utf-8 -*-


# ***************************************************
# * File        : GluontsDataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-11
# * Version     : 0.1.041120
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class GluontsDataset:

    def __init__(self, 
                 data = None, 
                 freq: str = None, 
                 start_datetime_str: str = None, 
                 predict_length: int = None) -> None:
        self.data = data
        self.freq = freq  # "1H"
        self.start_datetime_str = start_datetime_str  # "01-01-2019"
        self.predict_length = predict_length  # 24

    def get_train_test(self, data_name: str = None, target: str = None, windows: int = None):
        """
        gluonts datasets
        """
        if self.data is None:
            dataset = get_dataset(data_name)
            train_ds = dataset.train
            test_ds = dataset.test
        elif isinstance(self.data, np.ndarray):  # np.random.normal(size = (N, T))
            start = pd.Period(self.start_datetime_str, freq = self.freq)
            train_ds = ListDataset([{"target": x, "start": start} for x in self.data[:, :-self.predict_length]], freq = self.freq)
            test_ds = ListDataset([{"target": x, "start": start} for x in self.data], freq = self.freq)
        elif isinstance(self.data, pd.DataFrame):
            # data
            dataset = PandasDataset(self.data, target = target)
            # data split
            train_ds, test_ds = split(dataset, offset = - (self.predict_length * windows))
            test_ds = test_ds.generate_instances(prediction_length = self.predict_length, windows = windows)
        return train_ds, test_ds

    def get_long_data(self, url, target_name: str, item_id_col_name: str):
        df = pd.read_csv(url, index_col = 0, parse_dates = True)
        ds = PandasDataset.from_long_dataframe(df, target = target_name, item_id = item_id_col_name) 
        return ds
    
    def get_wide_data(self, url):
        df = pd.read_csv(url, index_col = 0, parse_dates = True)
        ds = PandasDataset(dict(df))
        return ds

    def train_plot(self, train_ds):
        # train series
        entry = next(iter(train_ds))
        train_series = to_pandas(entry)
        # plot
        train_series.plot()
        plt.grid(which = "both")
        plt.legend(["train series"], loc = "upper left")
        plt.show()

    def test_plot(self, train_ds, test_ds):
        # train series
        entry = next(iter(train_ds))
        train_series = to_pandas(entry)
        # test series
        entry = next(iter(test_ds))
        test_series = to_pandas(entry)
        # plot
        test_series.plot()
        plt.axvline(train_series.index[-1], c = "red")
        plt.legend(["test series"], loc = "upper left")
        plt.show()




# 测试代码 main 函数
def main():
    # ------------------------------
    # build in dataset
    # ------------------------------
    # gluonts_dataset = GluontsDataset()
    # train_ds, test_ds = gluonts_dataset.get_train_test(data_name = "m4_hourly")
    # gluonts_dataset.train_plot(train_ds)
    # gluonts_dataset.test_plot(train_ds, test_ds)
    # ------------------------------
    # custom dataset
    # ------------------------------
    gluonts_dataset = GluontsDataset(
        data = np.random.normal(size = (10, 100)),
        freq = "1H",
        start_datetime_str = "01-01-2019",
        predict_length = 24
    )
    train_ds, test_ds = gluonts_dataset.get_train_test()
    gluonts_dataset.train_plot(train_ds)
    gluonts_dataset.test_plot(train_ds, test_ds)
    # ------------------------------
    # 
    # ------------------------------
    # df = pd.read_csv(
    #     "https://raw.githubusercontent.com/AileenNielsen/"
    #     "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
    #     index_col = 0,
    #     parse_dates = True,
    # )
    # gluonts_dataset = GluontsDataset(data = df, predict_length = 36)
    # train_ds, test_ds = gluonts_dataset.get_train_test(target = "#Passengers", windows = 1)
    # gluonts_dataset.train_plot(train_ds)
    # gluonts_dataset.test_plot(train_ds, test_ds)

if __name__ == "__main__":
    main()
