# -*- coding: utf-8 -*-


# ***************************************************
# * File        : DeepAR.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-10
# * Version     : 0.1.041023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.mx import DeepAREstimator, Trainer
from gluonts.dataset.split import split

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DeepAR(object):

    def __init__(self, predict_length: int, freq: str, epochs: int = 5, windows: int = 3) -> None:
        self.predict_length = predict_length
        self.freq = freq
        self.epochs = epochs
        self.windows = windows

    def train(self, train_data):
        """
        model training
        """
        self.deepar = DeepAREstimator(
            prediction_length = self.predict_length,
            freq = self.freq,
            trainer = Trainer(epochs = self.epochs),
        ).train(train_data)

    def forecast(self, test_data):
        """
        model forecasting
        """
        self.forecasts = list(self.deepar.predict(test_data.input))

    def plot_forecast(self, raw_data):
        """
        raw and forecasting series plotting
        """
        raw_data.plot(color = "black")
        colors = ["green", "blue", "purple"]
        for color, forecast in zip(colors[:self.windows], self.forecasts):
            forecast.plot(color = f"tab:{color}")
        plt.legend(["True values"], loc = "upper left", fontsize = "xx-large")
        plt.show()




# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    from gluonts_utils.GluontsDataset import GluontsDataset
    import pandas as pd 

    df = pd.read_csv(
        "https://raw.githubusercontent.com/AileenNielsen/"
        "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
        index_col = 0,
        parse_dates = True,
    )
    gluont_dataset = GluontsDataset(data = df, predict_length = 36)
    train_data, test_data = gluont_dataset.get_train_test(target = "#Passengers", windows = 1)
    # ------------------------------
    # model
    # ------------------------------
    deepar = DeepAR(predict_length = 36, freq = "M", epochs = 5, windows = 1)
    deepar.train(train_data)
    deepar.forecast(test_data)
    # reslut plot
    deepar.plot_forecast(df["#Passengers"])

if __name__ == "__main__":
    main()
