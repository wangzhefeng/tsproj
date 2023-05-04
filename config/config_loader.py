# -*- coding: utf-8 -*-

# ***************************************************
# * File        : cfg_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-06-29
# * Version     : 0.1.062914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import os
from typing import Dict
import yaml


def load_yaml(file_name):
    with open(file_name, 'r', encoding = "utf-8") as infile:
        return yaml.load(
            infile, 
            Loader = yaml.FullLoader
        )


def get_params(yaml_path: str) -> Dict:
    """
    读取项目配置参数

    Returns:
        Dict: 项目配置参数
    """
    # 配置文件读取
    cfg_dir = os.path.dirname(__file__)
    # 项目配置 yaml 文件
    cfg_params = load_yaml(os.path.join(cfg_dir, yaml_path))

    return cfg_params




# 测试代码 main 函数
def main(): 
    sys_cfg_path = "./sys_cfg.yaml"
    cfg_params = get_params(sys_cfg_path)
    print(cfg_params)

if __name__ == "__main__":
    main()
