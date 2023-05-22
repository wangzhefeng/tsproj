# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pydebug.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-21
# * Version     : 0.1.052112
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def vector_dot(v1, v2):
    sum = 0
    for e1, e2 in zip(v1, v2):
        sum += e1 + e2
    
    return sum




# 测试代码 main 函数
def main():
    v1 = np.random.rand(10)
    v2 = np.random.rand(10)
    sum = vector_dot(v1, v2)
    print(sum)

if __name__ == "__main__":
    main()
