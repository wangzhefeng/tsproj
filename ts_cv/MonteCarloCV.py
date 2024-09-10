# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MonteCarloCV.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-10
# * Version     : 0.1.091017
# * Description : 时间序列蒙特卡洛交叉验证
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from typing import List, Generator

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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




# 测试代码 main 函数
def main():
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
