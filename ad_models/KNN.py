# -*- coding: utf-8 -*-


# ***************************************************
# * File        : KNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-30
# * Version     : 0.1.033022
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyod.models.knn import KNN


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


X_train = None

# 初始化检测器 clf
clf = KNN(method = "mean", n_neighbors = 3)
clf.fit(X_train)

# 返回训练数据上的分类标签(0: 正常值, 1:异常值)
y_train_pred = clf.labels_

# 返回训练数据上的异常值(分数越大越异常)
y_train_scores = clf.decision_scores_




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
