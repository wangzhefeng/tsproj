# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-10-29
# * Version     : 0.1.102919
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.datasets import UCR_UEA_datasets


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# ------------------------------
# 
# ------------------------------
my_first_time_series = [1, 3, 4, 2]
formatted_time_series = to_time_series(my_first_time_series)
print(formatted_time_series.shape)  # (4, 1)
print(formatted_time_series)


# ------------------------------
# 
# ------------------------------
my_first_time_series = [1, 3, 4, 2]
my_second_time_series = [1, 2, 4, 2]
formatted_dataset = to_time_series_dataset([
    my_first_time_series, 
    my_second_time_series
])
print(formatted_dataset.shape)  ## (2, 4, 1)
print(formatted_dataset)

my_third_time_series = [1, 2, 4, 2, 2]
formatted_dataset = to_time_series_dataset([
    my_first_time_series,
    my_second_time_series,
    my_third_time_series,
])
print(formatted_dataset.shape)  # (3, 5, 1)
print(formatted_dataset)



# ------------------------------
# 
# ------------------------------
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
print(X_train.shape)
print(y_train.shape)
print(X_train)









# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

