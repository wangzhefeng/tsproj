# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lib_FeatureTools.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031900
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import featuretools as ft

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
data = ft.demo.load_mock_customer()

# transactions dataframe
transactions_df = data["transactions"]\
    .merge(data["sessions"])\
    .merge(data["customers"])
transactions_df.sample(10)

# products dataframe
products_df = data["products"]
print(products_df)

# create an EntitySet
es = ft.EntitySet(id = "customer_data")
es = es.entity_from_dataframe(
    entity_id = "transactions",
    dataframe = transactions_df,
    index = "transaction_df",
    time_index = "transaction_time",
    variable_types = {
        "product_id": ft.variable_types.Categorical,
        "zip_code": ft.variable_types.ZIPCode
    },
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
