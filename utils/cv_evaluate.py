# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TimeSeriesSplit.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-10
# * Version     : 0.1.091017
# * Description : 时间序列分割
# * Link        : SciKit-Learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
# *               时间序列蒙特卡洛交叉验证: https://towardsdatascience.com/monte-carlo-cross-validation-for-time-series-ed01c41e2995/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from itertools import chain
from typing import List, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples

from cv_plot import plot_cv_indices
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO
class KFoldCV:

    def __init__(self, **kwargs):
        self.target = kwargs.pop("target", None)
        self.date_col = kwargs.pop("date_col", None)
        self.date_init = kwargs.pop("date_init", None)
        self.date_final = kwargs.pop("date_final", None)

        if kwargs:
            raise TypeError(f"Invalid parameters passed: {str(kwargs)}")
        
        if ((self.target is None) | (self.date_col is None) | (self.date_init is None) | (self.date_final is None)):
            raise TypeError("Incomplete inputs")

    def _train_test_split_time(self, X):
        n_arrays = len(X)
        if n_arrays == 0:
            raise ValueError("At least one array required as input")
        
        for i in range(self.date_init, self.date_final):
            train = X[X[self.date_col] < i]
            val = X[X[self.date_col] == i]
            X_train, X_test = train.drop([self.target], axis = 1), val.drop([self.target], axis = 1)
            y_train, y_test = train[self.target].values, val[self.target].values
        
            yield X_train, X_test, y_train, y_test

    def split(self, X):
        cv_t = self._train_test_split_time(X)
        return chain(cv_t)


class MonteCarloCV(_BaseKFold):
    """
    时间序列蒙特卡洛交叉验证
    """
 
    def __init__(self, 
                 n_splits: int, 
                 train_size: float, 
                 test_size: float, 
                 gap: int = 0):
        """
        Monte Carlo Cross-Validation
 
        Holdout applied in multiple testing periods
        Testing origin (time-step where testing begins) is randomly chosen according to a monte carlo simulation
 
        Parameters
        ----------
        n_splits: (int) Number of monte carlo repetitions in the procedure
        train_size: (float) Train size, in terms of ratio of the total length of the series
        test_size: (float) Test size, in terms of ratio of the total length of the series
        gap: (int) Number of samples to exclude from the end of each train set before the test set.
        """
        self.n_splits = n_splits
        self.n_samples = -1
        self.gap = gap
        self.train_size = train_size
        self.test_size = test_size
        self.train_n_samples = 0
        self.test_n_samples = 0 
        self.mc_origins = []
 
    def split(self, X, y = None, groups = None) -> Generator:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        self.n_samples = _num_samples(X)
 
        self.train_n_samples = int(self.n_samples * self.train_size) - 1
        self.test_n_samples = int(self.n_samples * self.test_size) - 1
 
        # Make sure we have enough samples for the given split parameters
        if self.n_splits > self.n_samples:
            raise ValueError(
                f'Cannot have number of folds={self.n_splits} greater'
                f' than the number of samples={self.n_samples}.'
            )
        if self.train_n_samples - self.gap <= 0:
            raise ValueError(
                f'The gap={self.gap} is too big for number of training samples'
                f'={self.train_n_samples} with testing samples={self.test_n_samples} and gap={self.gap}.'
            )
 
        indices = np.arange(self.n_samples)
        selection_range = np.arange(self.train_n_samples + 1, self.n_samples - self.test_n_samples - 1)
 
        self.mc_origins = np.random.choice(
            a = selection_range,
            size = self.n_splits,
            replace = True
        )
        for origin in self.mc_origins:
            if self.gap > 0:
                train_end = origin - self.gap + 1
            else:
                train_end = origin - self.gap
            train_start = origin - self.train_n_samples - 1
            test_end = origin + self.test_n_samples

            yield (
                indices[train_start:train_end],
                indices[origin:test_end],
            )
 
    def get_origins(self) -> List[int]:
        return self.mc_origins


class Rolling_Windows:
    
    def __init__(self, horizon: int=2, window_len: int=5, num_window: int=5, drop_last: bool=True):
        self.horizon = horizon
        self.window_len = window_len
        self.num_window = num_window
        self.drop_last = drop_last
    
    def _evaluate_split_index(self, window: int):
        """
        数据分割索引构建
        """
        test_end    = -1         + (-self.horizon) * (window - 1)
        test_start  = test_end   + (-self.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.window_len) + 1

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        logger.info(f"debug::split indexes: train_start:train_end: {train_start}:{train_end}")
        logger.info(f"debug::split indexes: test_start:test_end: {test_start}:{test_end}")
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            Y_test = data_Y.iloc[test_start:]
        else:
            X_test = data_X.iloc[test_start:test_end+1]
            Y_test = data_Y.iloc[test_start:test_end+1]

        return X_train, Y_train, X_test, Y_test

    def _evaluate_split_2(self, data_X, data_Y):
        """
        训练、测试数据集分割
        """
        tscv = TimeSeriesSplit(n_splits=self.num_window, max_train_size=self.window_len - self.horizon, test_size=self.horizon, gap=0)
        plot_cv_indices(tscv, n_splits = self.num_window, X=data_X, y=data_Y)
        for i, (train_index, test_index) in enumerate(tscv.split(data_X, data_Y, groups=None)):
            
            if self.drop_last and len(train_index) < self.window_len - self.horizon:
                continue
            logger.info(f"debug::Fold {i}: {train_index} {test_index}")
            X_train = data_X.iloc[train_index]
            Y_train = data_Y.iloc[train_index]
            X_test = data_X.iloc[test_index]
            Y_test = data_Y.iloc[test_index]
            logger.info(f"debug::X_train: \n{X_train}")
            logger.info(f"debug::Y_train: \n{Y_train}")
            logger.info(f"debug::X_test: \n{X_test}")
            logger.info(f"debug::Y_test: \n{Y_test}")


class Expanding_Windows:

    def __init__(self, horizon: int=2, min_train_window_len: int=5, num_window: int=5, drop_last: bool=True):
        self.horizon = horizon
        self.min_train_window_len = min_train_window_len
        self.num_window = num_window
        self.drop_last = drop_last
    
    def _evaluate_split_index(self, window: int):
        """
        数据分割索引构建
        """
        test_end = -1 + (-self.horizon) * (window - 1)
        test_start = test_end   + (-self.horizon) + 1
        train_end = test_start
        train_start = 0

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        logger.info(f"debug::split indexes: train_start:train_end: {train_start}:{train_end}")
        logger.info(f"debug::split indexes: test_start:test_end: {test_start}:{test_end}")
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            Y_test = data_Y.iloc[test_start:]
        else:
            X_test = data_X.iloc[test_start:test_end+1]
            Y_test = data_Y.iloc[test_start:test_end+1]

        return X_train, Y_train, X_test, Y_test

    def _evaluate_split_2(self, data_X, data_Y):
        """
        训练、测试数据集分割
        """
        tscv = TimeSeriesSplit(n_splits=self.num_window, max_train_size=None, test_size=self.horizon, gap=0)
        plot_cv_indices(tscv, n_splits = self.num_window, X=data_X, y=data_Y)
        for i, (train_index, test_index) in enumerate(tscv.split(data_X, data_Y, groups=None)):
            if self.drop_last and len(train_index) < self.min_train_window_len:
                continue
            logger.info(f"debug::Fold {i}: {train_index} {test_index}")
            X_train = data_X.iloc[train_index]
            Y_train = data_Y.iloc[train_index]
            X_test = data_X.iloc[test_index]
            Y_test = data_Y.iloc[test_index]
            logger.info(f"debug::X_train: \n{X_train}")
            logger.info(f"debug::Y_train: \n{Y_train}")
            logger.info(f"debug::X_test: \n{X_test}")
            logger.info(f"debug::Y_test: \n{Y_test}")




# 测试代码 main 函数
def main():
    # data 1
    # X = np.array([
    #     [1, 11],
    #     [2, 12],
    #     [3, 13],
    #     [4, 14],
    #     [5, 15],
    #     [6, 16],
    #     [7, 17],
    #     [8, 18],
    #     [9, 19],
    #     [10, 20],
    #     [11, 21],
    #     [12, 22],
    # ])
    # y = np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
    
    # data 2
    df = pd.DataFrame({
        "ds": pd.date_range("2025-06-01", periods=12, freq="M"),
        "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "X2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "y": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    })
    df = df.set_index("ds")
    X = df[["X1", "X2"]]
    y = df["y"]
    logger.info(f"X: \n{X}")
    logger.info(f"y: \n{y}")
    
    # ------------------------------
    # Rolling Windows
    # ------------------------------
    logger.info(f"{'-' * 40}")
    logger.info(f"Rolling Windows")
    logger.info(f"{'-' * 40}")
    num_window = 5
    # rolling window
    rolling_windows = Rolling_Windows(horizon=2, window_len=7, num_window=num_window)
    # method 1
    for window in range(1, num_window+1):
        X_train, Y_train, X_test, Y_test = rolling_windows._evaluate_split(X, y, window=window)
        if len(X_train) < rolling_windows.window_len - rolling_windows.horizon:
            continue
        logger.info(f"X_train: \n{X_train}")
        logger.info(f"Y_train: \n{Y_train}")
        logger.info(f"X_test: \n{X_test}")
        logger.info(f"Y_test: \n{Y_test}")
    # method 2
    rolling_windows._evaluate_split_2(X, y)
    
    # ------------------------------
    # Expanding Windows
    # ------------------------------
    logger.info(f"{'-' * 40}")
    logger.info(f"Expanding Windows")
    logger.info(f"{'-' * 40}")
    num_window = 5
    # expanding windows
    expanding_windows = Expanding_Windows(horizon=2, min_train_window_len=5, num_window=num_window)
    # method 1
    for window in range(1, num_window+1):
        X_train, Y_train, X_test, Y_test = expanding_windows._evaluate_split(X, y, window)
        if len(X_train) < expanding_windows.min_train_window_len:
            continue
        logger.info(f"X_train: \n{X_train}")
        logger.info(f"Y_train: \n{Y_train}")
        logger.info(f"X_test: \n{X_test}")
        logger.info(f"Y_test: \n{Y_test}")
    # method 2
    expanding_windows._evaluate_split_2(X, y)

    # ------------------------------
    # Monte Carlo Cross-Validation
    # ------------------------------
    logger.info(f"{'-' * 40}")
    logger.info(f"Monte Carlo Cross-Validation")
    logger.info(f"{'-' * 40}")
    num_window = 5
    # 数据分割
    mccv = MonteCarloCV(n_splits = num_window, train_size = 0.42, test_size = 0.25, gap = 0)
    plot_cv_indices(mccv, n_splits = num_window, X=X, y=y)
    for train_index, test_index in mccv.split(X):
        logger.info(f"\nTRAIN: {train_index} \nTEST: {test_index}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
        logger.info(f"X_train: \n{X_train}")
        logger.info(f"Y_train: \n{Y_train}")
        logger.info(f"X_test: \n{X_test}")
        logger.info(f"Y_test: \n{Y_test}")

if __name__ == "__main__":
    main()
