# -*- coding: utf-8 -*-


# ***************************************************
# * File        : DBSCAN.py
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
from sklearn.cluster import DBSCAN


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



# 数据
X = np.array(
    [[1, 2],
     [2, 2],
     [2, 3],
     [8, 7],
     [8, 8],
     [25, 80]]
)

# 聚类
clustering = DBSCAN(eps = 3, min_samples = 2).fit(X)
clustering.lables_

"""
array([ 0,  0,  0,  1,  1, -1])
# 0，,0，,0：表示前三个样本被分为了一个群
# 1, 1：中间两个被分为一个群
# -1：最后一个为异常点，不属于任何一个群
"""




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
