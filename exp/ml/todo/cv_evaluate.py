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
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class KFoldCV(object):

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
    
    def __init__(self):
        pass
    
    def _evaluate_split_index(self, window: int):
        """
        数据分割索引构建
        """
        test_end    = -1         + (-self.args.horizon) * (window - 1)
        test_start  = test_end   + (-self.args.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.args.window_len) + 1

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 数据分割
        X_train = data_X.loc[train_start:train_end]
        Y_train = data_Y.loc[train_start:train_end]
        X_test = data_X.loc[test_start:test_end]
        Y_test = data_Y.loc[test_start:test_end]
        logger.info(f"debug::split indexes: train_start:train_end: {train_start}:{train_end}")
        logger.info(f"debug::split indexes: test_start:test_end: {test_start}:{test_end}")

        return X_train, Y_train, X_test, Y_test


def expanding_windows():
    pass





# 测试代码 main 函数
def main():
    # ------------------------------
    # Expanding Windows
    # ------------------------------
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    X = np.array([
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
    ])
    y = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
    print(X.shape, y.shape)
    
    tscv = TimeSeriesSplit(n_splits=3, max_train_size=None, test_size=2, gap=0)
    for i, (train_index, test_index) in enumerate(tscv.split(X, y, groups=None)):
        print(f"Fold {i}: {train_index} {test_index}")
    
    # ------------------------------
    # Monte Carlo Cross-Validation
    # ------------------------------
    # 数据构造
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples = 120)

    # 数据分割
    mccv = MonteCarloCV(
        n_splits = 5, 
        train_size = 0.6, 
        test_size = 0.1, 
        gap = 0
    )
    for train_index, test_index in mccv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    # 超参数调优
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    model = RandomForestRegressor()
    param_search = {
        'n_estimators': [10, 100]
    }
    gsearch = GridSearchCV(
        estimator = model, 
        cv = mccv, 
        param_grid = param_search
    )
    gsearch.fit(X, y)

if __name__ == "__main__":
    main()
