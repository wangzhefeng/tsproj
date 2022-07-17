# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_logging.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071721
# * Description : 数据日志查看
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def data_logging(data: pd.DataFrame, data_name: str):
    """
    查看数据结构、内容信息
    """
    print("=" * 40)
    print(f"{data_name}.head()")
    print("=" * 40)
    print(data.head())
    print("=" * 40)
    print(f"{data_name}.tail()")
    print("=" * 40)
    print(data.tail())
    print("=" * 40)
    print(f"{data_name}.info()")
    print("=" * 40)
    print(data.info())



__all__ = [
    data_logging,
]


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

