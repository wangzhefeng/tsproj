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
color_pal = [
    "#F8766D", "#D39200", "#93AA00",
    "#00BA38", "#00C19F", "#00B9E3",
    "#619CFF", "#DB72FB"
]
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data load
df = pd.read_csv("PJME_hourly.csv", index_col=[0], parse_dates=[0])
df.columns = ["y"]
df.index.name = "ds"
# data visual
df.plot(style='.', figsize=(15,5), color=color_pal[1], title='PJM East')
plt.show()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
