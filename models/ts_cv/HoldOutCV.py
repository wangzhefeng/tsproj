# -*- coding: utf-8 -*-

# ***************************************************
# * File        : HoldOutCV.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-10
# * Version     : 0.1.091017
# * Description : 时间序列 Hold-Out 交叉验证
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sklearn.model_selection import train_test_split

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO
tts = train_test_split()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
