# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LOF.py
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
from sklearn.neighbors import LocalOutlierFactor as LOF


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def LofOutlierDetection(data, neighbor = 50, dist_metric = "l1", contamination = "auto"):
    """
    基于LOF算法的单特征时序数据异常值检测
    Parameters:
        data: series list
        neighbor: 近邻数
        dist_metric:距离计算方法
        contamination: 异常值比例
    Returns:
        序列数据标签, -1为异常值, 1为非异常值
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = LOF(
        n_neighbors = neighbor, 
        metric = dist_metric, 
        contamination = contamination
    )
    res = clf.fit_predict(s_data)
    neg_outlier_factor = clf.negative_outlier_factor_
    
    return res, neg_outlier_factor





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
