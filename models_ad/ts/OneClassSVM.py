# -*- coding: utf-8 -*-


# ***************************************************
# * File        : OneClassSVM.py
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

import numpy as np
from sklearn import svm


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def OneClassSvmOutlierDetection(data, kernel="rbf", gamma=0.1, nu=0.3):
    """
    基于OneClassSvm算法的单特征时序数据异常值检测
    Parameters:
        data: series list
        kernel:
        threshold: 异常值分数阈值
    Returns:
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
    error_data = np.asarray(data).reshape(-1, 1)

    # fit the model
    clf = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    y_pred = clf.fit_predict(error_data)
    n_error_outlier = y_pred[y_pred == -1].size

    return y_pred, n_error_outlier




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
