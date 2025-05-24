# -*- coding: utf-8 -*-


# ***************************************************
# * File        : IsolationForest.py
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
from sklearn.ensemble import IsolationForest


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def IsolationForestAD(data, max_samples = "auto", contamination = "auto", threshold = None):
    """
    基于 isolation forest 算法的单特征时序数据异常值检测

    Args:
        data (_type_): series list
        max_samples (str, optional): 参考sklearn文档. Defaults to "auto".
        contamination (str, optional): 参考sklearn文档. Defaults to "auto".
        threshold (_type_, optional): 异常值分数阈值. Defaults to None.

    Returns:
        _type_: 序列数据标签, -1 为异常值, 1 为非异常值
    
    Link:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    s_data = np.asarray(data).reshape(-1, 1)
    clf = IsolationForest(
        max_samples = max_samples, 
        contamination = contamination
    )
    if threshold is None:
        labels = clf.fit_predict(s_data)
        scores = clf.decision_function(s_data)
        outlier_index = np.argwhere(labels == -1).reshape(-1,)
        return labels, scores, outlier_index
    else:
        label_res = []
        scores = clf.score_samples(s_data)
        outlier_index = np.argwhere(labels == -1).reshape(-1,)
        for score_i in scores:
            if score_i < threshold:
                label_res.append(-1)
            else:
                label_res.append(1)
        
        return label_res, scores, outlier_index




# 测试代码 main 函数
def main():
    from sklearn.datasets import load_iris

if __name__ == "__main__":
    main()
