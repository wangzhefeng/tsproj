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


def load_json_config(config_filename):
    """
    读取项目配置参数

    Args:
        config_filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # json 配置文件所在路径
    cfg_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(cfg_dir, config_filename)
    # 读取项目配置 json文件
    with open(cfg_path, "r", encoding = "utf-8") as infile:
        cfg_params = json.load(infile)

    # 构建模型保存文件夹
    if not os.path.exists(cfg_params['model']['save_dir']): 
        os.makedirs(cfg_params['model']['save_dir'])

    return cfg_params




# 测试代码 main 函数
def main():
    configs = load_json_config("config_sp500_1.json")
    logger.info(configs)

if __name__ == "__main__":
    main()
