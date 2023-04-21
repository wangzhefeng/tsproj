# -*- coding: utf-8 -*-


# ***************************************************
# * File        : SimpleFeedForward.py
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
import json

import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
from gluonts.evaluation import (
    make_evaluation_predictions,
    Evaluator,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class SimpleFeedForward:

    def __init__(self, 
                 predict_length: int, 
                 learning_rate: float = 1e-3, 
                 num_batches: int = 100,
                 epochs: int = 5) -> None:
        self.predict_length = predict_length
        self.learning_rate = learning_rate
        self.num_batches = num_batches
        self.epochs = epochs

    def train(self, train_data):
        """
        model training
        """
        self.estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions = [10],
            prediction_length = self.predict_length,
            context_length = 100,
            trainer = Trainer(
                ctx = "cpu", 
                epochs = self.epochs, 
                learning_rate = self.learning_rate, 
                num_batches_per_epoch = self.num_batches,
            )
        )
        self.predictor = self.estimator.train(train_data)

    def forecast(self, test_data):
        """
        model forecasting
        """
        forecast_it, ts_it = make_evaluation_predictions(
            dataset = test_data,
            predictor = self.predictor,
            num_samples = 100
        )
        forecasts = list(forecast_it)  # [0]
        tss = list(ts_it)  # [0]

        evaluator = Evaluator(quantiles = [0.1, 0.5, 0.9])
        self.agg_metrics, self.item_metrics = evaluator(tss, forecasts)
        print(json.dumps(self.agg_metrics, indent = 4))
        print(self.item_metrics.head())

        return tss, forecasts
    
    def plot_prob_forecasts(self, ts_entry, forecast_entry):
        plot_length = 150
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        fig, ax = plt.subplots(1, 1, figsize = (10, 7))
        ts_entry[-plot_length:].plot(ax = ax)  # plot the time series
        forecast_entry.plot(prediction_intervals = prediction_intervals, color = "g")
        plt.grid(which = "both")
        plt.legend(legend, loc = "upper left")
        plt.show()
    
    def plot_metrics(self):
        self.item_metrics.plot(x = "MSIS", y = "MASE", kind = "scatter")
        plt.grid(which = "both")
        plt.show()




# 测试代码 main 函数
def main():
    # data
    import numpy as np
    from gluonts_utils.GluontsDataset import GluontsDataset
    gluont_dataset = GluontsDataset(
        data = np.random.normal(size = (10, 100)), 
        freq = "H", 
        start_datetime_str = "01-01-2019", 
        predict_length = 24
    )
    train_data, test_data = gluont_dataset.get_train_test()
    # model
    model = SimpleFeedForward(
        predict_length = 48,
        learning_rate = 1e-3,
        num_batches = 100,
        epochs = 5,
    )
    model.train(train_data)
    # model forecast
    tss, forecasts = model.forecast(test_data)
    model.plot_prob_forecasts(tss[0], forecasts[0])
    model.plot_metrics()

if __name__ == "__main__":
    main()
