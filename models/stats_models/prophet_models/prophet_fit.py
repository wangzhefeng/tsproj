# -*- coding: utf-8 -*-

# ***************************************************
# * File        : prophet_fit.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091118
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
