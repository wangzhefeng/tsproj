# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yaml_config_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-31
# * Version     : 0.1.053122
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
from typing import Dict
import yaml

from loguru import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_yaml(yaml_path: str):
    """
    读取项目配置参数

    Args:
        yaml_path (str): _description_

    Returns:
        Dict: 项目配置参数
    """
    # 配置文件读取
    cfg_dir = os.path.dirname(__file__)
    # 项目配置 yaml 文件
    with open(os.path.join(cfg_dir, yaml_path), 'r', encoding = "utf-8") as infile:
        cfg_params = yaml.load(
            infile, 
            Loader = yaml.FullLoader
        )
    
    return cfg_params




# 测试代码 main 函数
def main(): 
    sys_cfg_path = "config_wind.yaml"
    cfg_params = load_yaml(sys_cfg_path)
    logger.info(cfg_params)

if __name__ == "__main__":
    main()
