# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-13
# * Version     : 0.1.111300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pmdarima as pm


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


x = pm.c(1, 2, 3, 4, 5, 6, 7)
print(x)

acf = pm.acf(x)
print(acf)

pm.plot_acf(x)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

