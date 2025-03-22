# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-22
# * Version     : 1.0.032213
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]







# 测试代码 main 函数
def main():
    import pandas as pd

    df = pd.DataFrame({
        "ds": pd.date_range(start="2022-01-01", end="2022-12-31", freq="D"),
        "y": range(1, 366)
    })
    print(df.head())
    del df["y"]
    print(df.head())

if __name__ == "__main__":
    main()
