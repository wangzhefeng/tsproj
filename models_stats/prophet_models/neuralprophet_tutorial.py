# -*- coding: utf-8 -*-

# ***************************************************
# * File        : neuralprophet_tutorial.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091401
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd
import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# data
df = pd.read_csv("https://github.com/ourownstory/neuralprophet-data/raw/main/kaggle-energy/datasets/tutorial01.csv")
print(df.head())
print(df.shape)
df.plot(x = "ds", y = "y", figsize = (15, 5))
plt.show()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
