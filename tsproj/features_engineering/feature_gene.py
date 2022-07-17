# -*- coding: utf-8 -*-


# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071720
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from typing import List

import pandas as pd


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def gen_time_features(df: pd.DataFrame, 
                      datetime_format: str, 
                      datetime_is_index: bool = False, 
                      datetime_name: str = None, 
                      features: List = None,
                      ) -> pd.DataFrame:
    """
    时间特征提取

    Args:
        data ([type]): 时间序列
        datetime_format ([type]): 时间特征日期时间格式
        datetime_is_index (bool, optional): 时间特征是否为索引. Defaults to False.
        datetime_name ([type], optional): 时间特征名称. Defaults to None.
        target_name ([type], optional): 目标特征名称. Defaults to False.
        features: 最后返回的特征名称列表
    """
    data = df.copy()
    if datetime_is_index:
        data["DT"] = data.index
        data["DT"] = pd.to_datetime(data["DT"], format = datetime_format)
    else:
        data[datetime_name] = pd.to_datetime(data[datetime_name], format = datetime_format)
        data["DT"] = data[datetime_name]
    data["year"] = data["DT"].apply(lambda x: x.year)
    data["quarter"] = data["DT"].apply(lambda x: x.quarter)
    data["month"] = data["DT"].apply(lambda x: x.month)
    data["day"] = data["DT"].apply(lambda x: x.day)
    data["hour"] = data["DT"].apply(lambda x: x.hour)
    data["minute"] = None
    data["second"] = None
    data["dow"] = data["DT"].apply(lambda x: x.dayofweek)
    data["doy"] = data["DT"].apply(lambda x: x.dayofyear)
    data["woy"] = data["DT"].apply(lambda x: x.weekofyear)
    data["year_start"] = data["DT"].apply(lambda x: x.is_year_start)
    data["year_end"] = data["DT"].apply(lambda x: x.is_year_end)
    data["quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)
    data["quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)
    data["month_start"] = data["DT"].apply(lambda x: x.is_month_start)
    data["month_end"] = data["DT"].apply(lambda x: x.is_month_end)
    def applyer(row):
        """
        判断是否是周末
        """
        if row == 5 or row == 6:
            return 1
        else:
            return 0
    data["weekend"] = data['dow'].apply(applyer)
    del data["DT"]
    if features is None:
        selected_features = data
    else:
        selected_features = data[features]
    return selected_features



__all__ = [
    gen_time_features,
]


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()





