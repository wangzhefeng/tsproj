# -*- coding: utf-8 -*-

# ***************************************************
# * File        : time_col_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-31
# * Version     : 1.0.053123
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


def time_col_checkout(df, time_col, time_format='%Y-%m-%d %H:%M:%S'):
    """
    检查时间序列的时间戳列是否存在缺失
    """
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], format=time_format)
    df_copy["delta"] = df_copy[time_col] - df_copy[time_col].shift(1)
    results = df_copy["delta"].value_counts()

    return results


def time_col_distinct(df, time_col: str, time_format='%Y-%m-%d %H:%M:%S'):
    """
    对时间序列的时间戳列进行去重
    """
    # 转换时间戳类型
    df[time_col] = pd.to_datetime(df[time_col], format=time_format)
    # 去除重复时间戳
    df.drop_duplicates(subset=time_col, keep="last", inplace=True, ignore_index=True) 

    return df




# 测试代码 main 函数
def main():
    df = pd.DataFrame({
        "time": [
            "2025-01-01 00:00:00", 
            "2025-01-02 00:00:00", 
            "2025-01-02 00:00:00", 
            "2025-01-04 00:00:00", 
            "2025-01-05 00:00:00"
        ],
        "value": [1, 2, 3, 4, 5],
    })
    res = time_col_checkout(df, time_col="time")
    print(res)

    df = time_col_distinct(df, time_col="time")
    print(df)
 
if __name__ == "__main__":
    main()
