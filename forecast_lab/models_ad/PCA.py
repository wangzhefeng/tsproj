# -*- coding: utf-8 -*-


# ***************************************************
# * File        : PCA.py
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
import pandas as pd
from sklearn.decomposition import PCA


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


pca = PCA()
pca.fit(centered_training_data)
transformed_data = pca.transform(training_data)
y = transformed_data

# 计算异常分数
lambdas = pca.singular_values_
M = ((y*y)/lambdas)

# 前k个特征向量和后r个特征向量
q = 5
print("Explained variance by first q terms: ", sum(pca.explained_variance_ratio_[:q]))
q_values = list(pca.singular_values_ < .2)
r = q_values.index(True)

# 对每个样本点进行距离求和的计算
major_components = M[:,range(q)]
minor_components = M[:,range(r, len(features))]
major_components = np.sum(major_components, axis=1)
minor_components = np.sum(minor_components, axis=1)

# 人为设定c1、c2阈值
components = pd.DataFrame({'major_components': major_components, 
                               'minor_components': minor_components})
c1 = components.quantile(0.99)['major_components']
c2 = components.quantile(0.99)['minor_components']

# 制作分类器
def classifier(major_components, minor_components):  
    major = major_components > c1
    minor = minor_components > c2    
    return np.logical_or(major,minor)

results = classifier(major_components=major_components, minor_components=minor_components)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
