# -*- coding: utf-8 -*-

# ***************************************************
# * File        : visual.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-04
# * Version     : 0.1.010414
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def test_result_visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    fig = plt.figure(figsize=(12, 8))
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig(name, bbox_inches='tight')



# 测试代码 main 函数
def main():
   pass

if __name__ == "__main__":
   main()
