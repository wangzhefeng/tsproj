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
import warnings

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def test_result_visual(trues, preds=None, path='./pic/test.pdf'):
    """
    Results visualization
    """
    # 设置绘图风格
    # plt.style.use('ggplot')
    # 画布
    fig = plt.figure(figsize = (25, 8))
    # 创建折线图
    plt.plot(trues, label='Trues', linewidth=1)  # 实际值
    plt.plot(preds, label='Preds', linewidth=1, linestyle="--")  # 预测值
    # 增强视觉效果
    plt.legend()
    plt.xlabel("日期时间")
    plt.ylabel("Value")
    plt.title('实际值 vs 预测值')
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.show();




# 测试代码 main 函数
def main():
   pass

if __name__ == "__main__":
   main()
