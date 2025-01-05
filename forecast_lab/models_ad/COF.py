# -*- coding: utf-8 -*-


# ***************************************************
# * File        : COF.py
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
from sklearn.datasets import load_iris
from pyod.models.cof import COF


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


cof = COF(
    contamination = 0.06,  # 异常值所占的比例
    n_neighbors = 20,  # 近邻数量
)

iris = load_iris()
cof_label = cof.fit_predict(iris.values)
print(f"检测出的异常值数量为：{np.sum(cof_label == 1)}")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
