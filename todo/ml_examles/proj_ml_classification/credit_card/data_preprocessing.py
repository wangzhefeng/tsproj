# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_preprocessing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-27
# * Version     : 0.1.022714
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import numpy as np


def data_normalize(train_features, validation_features):
    mean = np.mean(train_features, axis = 0)
    std = np.std(train_features, axis = 0)
    train_features -= mean
    train_features /= std
    validation_features -= mean
    validation_features /= std

    return train_features, validation_features


# 测试代码 main 函数
def main():
    from data_load import data_load
    from data_generator import data_generator
    
    features, targets = data_load()

    (train_features, train_targets), \
    (validation_features, validation_targets) = data_generator(features, targets)
    
    train_features, validation_features = data_normalize(train_features, validation_features)


if __name__ == "__main__":
    main()

