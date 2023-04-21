# -*- coding: utf-8 -*-


# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import yaml


def load_yaml(file_name):
    with open(file_name, "r", encoding = "utf-8") as infile:
        file_data = infile.read()
        return yaml.load(file_data, Loader = yaml.FullLoader)


config_dir = os.path.dirname(__file__)
settings = load_yaml(os.path.join(config_dir, "config.yaml"))



def main():
    print(config_dir)
    print(settings)
    print(settings["PATH"]["data_path"])
    print(settings["PATH"]["model_path"])


if __name__ == "__main__":
    main()

