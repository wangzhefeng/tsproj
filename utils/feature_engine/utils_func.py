# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils_func.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-18
# * Version     : 0.1.071822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


__all__ = [
    "is_weekend",
]


# python libraries
import os
import sys


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def is_weekend(row: int) -> int:
    """
    判断是否是周末
    
    Args:
        row (int): 一周的第几天

    Returns:
        int: 0: 不是周末, 1: 是周末
    """
    if row == 5 or row == 6:
        return 1
    else:
        return 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

