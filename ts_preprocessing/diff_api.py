# -*- coding: utf-8 -*-

# ***************************************************
# * File        : diff_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090823
# * Description : 时间序列差分
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

"""
# 1 阶差分、1步差分
pandas.DataFrame.diff(periods = 1, axis = 0)

# 2 步差分
pandas.DataFrame.diff(periods = 2, axis = 0)

# k 步差分
pandas.DataFrame.diff(periods = k, axis = 0)

# -1 步差分
pandas.DataFrame.diff(periods = -1, axis = 0)
"""









# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
