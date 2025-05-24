# -*- coding: utf-8 -*-

# ***************************************************
# * File        : na_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-24
# * Version     : 1.0.052422
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

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def na_fill_with_ma(df: pd.DataFrame, col_with_na: str):
    """
    用移动平均值填充缺失值

    Args:
        df (pd.DataFrame): 数据集
        col_with_na (str): 需要进行缺失值处理的变量名

    Returns:
        _type_: _description_
    """
    moving_avg = df[col_with_na].rolling(window=5, min_periods=1).mean()
    df[col_with_na] = df.fillna(moving_avg)

    return df




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
