# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ToSupervised.py
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
from sklearn import base


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ToSupervised(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col, groupCol, numLags, dropna = False):
        self.col = col
        self.groupCol = groupCol
        self.numLags = numLags
        self.dropna = dropna

    def fit(self, X, y = None):
        self.X = X
        return self

    def transform(self, X):
        tmp = self.X.copy()
        for i in range(1, self.numLags + 1):
            tmp[str(i) + "_Week_Ago" + "_" + self.col] = tmp.groupby([self.groupCol])[self.col].shift(i)
        
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop = True)
        
        return tmp




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
