# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_preprocess.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-09
# * Version     : 1.0.120913
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
import math

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def calc_ups_active_power(df):
    """
    UPS本体 201、204 三相输出有功功率计算
    """
    df["A相输出有功功率"] = math.sqrt(3) * df["A相输出电流"] * (df["A相输出电压"] / 1000) * df["实际输出PF"]
    df["B相输出有功功率"] = math.sqrt(3) * df["B相输出电流"] * (df["B相输出电压"] / 1000) * df["实际输出PF"]
    df["C相输出有功功率"] = math.sqrt(3) * df["B相输出电流"] * (df["C相输出电压"] / 1000) * df["实际输出PF"]

    return df


def calc_cabinet_active_power(df, cabinet: str = "A01", line: str = "A1"):
    """
    机柜电功率计算
    """
    df[f"{cabinet}-{line}有功功率"] = df[f"{cabinet}-{line}电流"] * (df[f"{cabinet}-{line}电压"] / 1000) * df[f"{cabinet}-{line}功率因数"]

    return df




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
