# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ExpandingFeatures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042415
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ExpandingFeatures:
    
    def __init__(self, 
                 data, 
                 datetime_format: str = '%Y-%m-%d %H:%M:%S',
                 window_length: int = 7) -> None:
        self.data = data
        self.datetime_format = datetime_format
        self.window_length = window_length

    def features(self, raw_feature, new_feature):
        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"], format = self.datetime_format)
        self.data[new_feature] = self.data[raw_feature].expanding(2).mean()
        data_columns = ["Datetime", raw_feature, new_feature]
        self.data = self.data[data_columns]







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
