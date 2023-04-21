# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_normalize.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-27
# * Version     : 0.1.032722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") in sys.path:
    pass
else:
    sys.path.append(os.path.join(_path, ".."))

import numpy as np
from typing import Tuple
from config.config_loader import settings
from data_split import data_split


def data_normalize(train_array: np.ndarray, 
                   validation_array: np.ndarray, 
                   test_array: np.ndarray) -> Tuple[np.ndarray]:
    train_mean, train_std = train_array.mean(axis = 0), train_array.std(axis = 0)

    train_array = (train_array - train_mean) / train_std
    validation_array = (validation_array - train_mean) / train_std
    test_array = (test_array - train_mean) / train_std

    return train_array, validation_array, test_array




# 测试代码 main 函数
def main():
    from utils_data.data_loader import data_loader
    
    route_distances, speeds_array = data_loader()
    train_array, validation_array, test_array = data_split(
        speeds_array, 
        settings["DATA"]["train_size"],
        settings["DATA"]["validation_size"],
    )
    train_array, validation_array, test_array = data_normalize(
        train_array, 
        validation_array, 
        test_array,
    )
    print(f"train set size: {train_array.shape}")
    print(f"validation set size: {validation_array.shape}")
    print(f"test set size: {test_array.shape}")

    

    


if __name__ == "__main__":
    main()

