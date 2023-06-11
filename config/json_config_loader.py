# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052222
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
import json

from loguru import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_config(config_filename):
    # load config file
    config_dir = os.path.join(os.path.dirname(__file__), config_filename)
    configs = json.load(open(config_dir, 'r'))

    # config project
    if not os.path.exists(configs['model']['save_dir']): 
        os.makedirs(configs['model']['save_dir'])

    return configs




# 测试代码 main 函数
def main():
    configs = load_config("lstm/config_sp500_2.json")
    logger.info(configs)

if __name__ == "__main__":
    main()
