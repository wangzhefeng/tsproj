# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Naive_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-10
# * Version     : 1.0.091022
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sklearn.base import BaseEstimator

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Navie(BaseEstimator):
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return X["lag1"]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
