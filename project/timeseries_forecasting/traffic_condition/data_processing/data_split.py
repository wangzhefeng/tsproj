# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_split.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-27
# * Version     : 0.1.032718
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np
from typing import Tuple


def data_split(data_array: np.ndarray, 
               train_size: float, 
               validation_size: float) -> Tuple[np.ndarray]:
    """
    将数据分割为 train/validation/test 数据集

    Args:
        data_array (np.ndarray): ndarray of shape `(num_time_steps, num_routes)`
        train_size (float): A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        validation_size (float): A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.
    
    Returns:
        `train_array`, `val_array`、`test_array`
    """
    num_time_steps = data_array.shape[0]
    num_train, num_validation = (
        int(num_time_steps * train_size), 
        int(num_time_steps * validation_size)
    )
    train_array = data_array[:num_train]
    validation_array = data_array[num_train:(num_train + num_validation)]
    test_array = data_array[(num_train + num_validation):]
    
    return train_array, validation_array, test_array
    



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

