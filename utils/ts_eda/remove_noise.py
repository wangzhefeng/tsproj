# -*- coding: utf-8 -*-

# ***************************************************
# * File        : remove_noise.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-25
# * Version     : 1.0.052511
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


def remove_noise(df: pd.DataFrame, window: int = 3):
    """
    使用移动平均法去噪

    Args:
        df (pd.DataFrame): _description_
        window (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    df_smooth = df.values.rolling(window = window).mean()

    return df_smooth




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
