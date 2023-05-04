# -*- coding: utf-8 -*-


# ***************************************************
# * File        : example.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-15
# * Version     : 0.1.111500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from autots.datasets import load_monthly
from autots import AutoTS


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
df_long = load_monthly(long = True)
print(df_long.head())


# model
model = AutoTS(
    forecast_length = 3,
    frequency = "infer",
    ensemble = "simple",
    max_generations = 5,
    num_validations = 2,
)
model = model.fit(
    df_long, 
    date_col = "datetime",
    value_col = "value",
    id_col = "series_id",
)

# best model info
print(model)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

