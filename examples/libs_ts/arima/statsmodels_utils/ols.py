# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ols.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-16
# * Version     : 0.1.111623
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
data = sm.datasets.get_rdataset("Guerry", "HistData").data
print(data)

# regression model
results = smf.ols("Lottery ~ Literacy + np.log(Pop1831)", data = data).fit()
print(results.summary())
print(dir(results))
print(results.__doc__)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

