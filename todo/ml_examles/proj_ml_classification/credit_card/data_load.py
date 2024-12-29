# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-27
# * Version     : 0.1.022713
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import csv
import numpy as np
from config.config_loader import settings


def data_load():
    all_features = []
    all_targets = []
    with open(settings["PATH"]["data_path"]) as f:
        for i, line in enumerate(f):
            if i == 0:
                print("HEADER:", line.strip())
                continue # skip header
            fields = line.strip().split(",")
            all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
            all_targets.append([int(fields[-1].replace('"', ""))])
            if i == 1:
                print("EXAMPLE FEATURES:", all_features[-1])
    features = np.array(all_features, dtype = "float32")
    targets = np.array(all_targets, dtype = "uint8")
    print("features.shape:", features.shape)
    print("targets.shape:", targets.shape)

    return features, targets




# 测试代码 main 函数
def main():
    features, targets = data_load()


if __name__ == "__main__":
    main()

