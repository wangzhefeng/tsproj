# -*- coding: utf-8 -*-


# ***************************************************
# * File        : SOS.py
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

import pandas as pd
from sksos import SOS


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
iris = pd.read_csv("http://bit.ly/iris-csv")
X = iris.drop("Name", axis = 1).values

# model
detector = SOS()
iris["score"] = detector.predict(X)
iris.sort_values("score", ascending = False).head(10)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
