# -*- coding: utf-8 -*-


# ***************************************************
# * File        : TimeSeriesKFold.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-18
# * Version     : 0.1.031822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from itertools import chain


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

    


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
