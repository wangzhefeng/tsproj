# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-10
# * Version     : 0.1.121017
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

# data
from sktime.datasets import load_airline
from sktime.datasets import load_arrow_head

# model
from sktime.forecasting.base import ForecastingHorizon
# model forecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.regression.compose import ComposableTimeSeriesForestRegressor
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.annotation.adapters import PyODAnnotator

from pyod.models.iforest import IForest

# model selection
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import train_test_split

# model performance
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def Forecasting():
    # data
    y = load_airline()
    print(y)
    y_train, y_test = temporal_train_test_split(y)
    print(y_train.shape)
    print(y_test.shape)
    # 
    fh = ForecastingHorizon(y_test.index, is_relative = False)
    print(fh)
    # model
    forecaster = ThetaForecaster(sp = 12)
    forecaster.fit(y_train)
    # model predict
    y_pred = forecaster.predict(fh)
    print(y_pred)
    # model evaluate
    score = mean_absolute_percentage_error(y_test, y_pred)
    print(score)


def TimeSeriesClassification():
    # data
    X, y = load_arrow_head()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # model
    classifier = TimeSeriesClassification()
    classifier.fit(X_train, y_train)
    # model predict
    y_pred = classifier.predict(X_test)
    # model evaluate
    score = accuracy_score(y_test, y_pred)
    print(score)


def TimeSeriesRegression():
    pass


def TimeSeriesClustering():
    # data
    X, y = load_arrow_head()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # model
    k_means = TimeSeriesKMeans(n_clusters = 5, init_algorithm = "forgy", metric = "dtw")
    k_means.fit(X_train)
    # model evaluate
    plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)


def TimeSeriesAnnotation():
    # data
    y = load_airline()
    # model
    pyod_model = IForest()
    pyod_sktime_annotator = PyODAnnotator(pyod_model)
    pyod_sktime_annotator.fit(y)
    # mode evaluate
    annotated_series = pyod_sktime_annotator.predict(y)
    print(annotated_series)




# 测试代码 main 函数
def main():
    TimeSeriesClustering()

if __name__ == "__main__":
    main()

