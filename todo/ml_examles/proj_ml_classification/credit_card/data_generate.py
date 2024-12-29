# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_generator.py
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
import numpy as np
from config.config_loader import settings


def data_generator(features, targets):
    num_val_samples = int(len(features) * settings["DATA"]["validation_split"])
    train_features = features[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    validation_features = features[-num_val_samples:]
    validation_targets = targets[-num_val_samples:]

    print("Number of training samples:", len(train_features))
    print("Number of validation samples:", len(validation_features))

    

    return (train_features, train_targets), (validation_features, validation_targets)


def weight_generate(train_targets):
    """
    analyze class imbalance in the targets

    Args:
        train_targets (_type_): _description_

    Returns:
        _type_: _description_
    """
    counts = np.bincount(train_targets[:, 0])
    print(
        "Number of positive samples in training data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(train_targets)
        )
    )
    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]

    return weight_for_0, weight_for_1


# 测试代码 main 函数
def main():
    from data_load import data_load

    features, targets = data_load()
    
    (trian_features, train_targets), \
    (validations_features, validation_targets) = data_generator(features, targets)
    
    weight_for_0, weight_for_1 = weight_generate(train_targets)


if __name__ == "__main__":
    main()

