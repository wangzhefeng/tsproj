# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgbm_power_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.random_seed import set_seed_ml
from utils.log_util import logger


def args_parse():
    pass


def run(args):
    pass




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed_ml(seed = 2025)
    # 参数解析
    args = args_parse()
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
