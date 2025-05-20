# -*- coding: utf-8 -*-

# ***************************************************
# * File        : filter_integer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-14
# * Version     : 1.0.051414
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
import re

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def filter_zh(input_str):
    """
    筛选字符串中的中文字符
    """
    output_str = re.compile(r"[\u4e00-\u9fff]+").findall(input_str)

    return output_str


def filter_integer(input_str):
    """
    筛选字符串中的整数
    """
    output_str = re.compile(r"\d+").findall(input_str)

    return output_str


def filter_number(input_str):
    """
    筛选字符串中的数字
    """
    output_str = re.findall(r"\d+\.?\d*", input_str)

    return output_str




# 测试代码 main 函数
def main():
    input_str = "5min"
    print(filter_zh(input_str))
    print(filter_integer(input_str))
    print(filter_number(input_str))

if __name__ == "__main__":
    main()
