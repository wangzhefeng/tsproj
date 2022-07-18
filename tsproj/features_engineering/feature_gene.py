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

from utils_func import is_weekend


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    
    # TODO 日期时间特征处理
    if datetime_is_index:
        data["DT"] = data.index
        data["DT"] = pd.to_datetime(data["DT"], format = datetime_format)
    else:
        data[datetime_name] = pd.to_datetime(data[datetime_name], format = datetime_format)
        data["DT"] = data[datetime_name]
    
    # 时间特征提取
    data["year"] = data["DT"].apply(lambda x: x.year)  # 年
    data["quarter"] = data["DT"].apply(lambda x: x.quarter)  # 季度
    data["month"] = data["DT"].apply(lambda x: x.month)  # 月
    data["day"] = data["DT"].apply(lambda x: x.day)  # 日
    data["hour"] = data["DT"].apply(lambda x: x.hour)  # 时
    data["minute"] = None  # 分
    data["second"] = None  # 秒
    data["dow"] = data["DT"].apply(lambda x: x.dayofweek)  # 一周的第几天
    data["doy"] = data["DT"].apply(lambda x: x.dayofyear)  # 一年的第几天
    data["woy"] = data["DT"].apply(lambda x: x.weekofyear)  # 一年的第几周
    data["year_start"] = data["DT"].apply(lambda x: x.is_year_start)  # 是否年初
    data["year_end"] = data["DT"].apply(lambda x: x.is_year_end)  # 是否年末
    data["quarter_start"] = data["DT"].apply(lambda x: x.is_quarter_start)  # 是否季度初
    data["quarter_end"] = data["DT"].apply(lambda x: x.is_quarter_end)  # 是否季度末
    data["month_start"] = data["DT"].apply(lambda x: x.is_month_start)  # 是否月初
    data["month_end"] = data["DT"].apply(lambda x: x.is_month_end)  # 是否月末
    data["weekend"] = data['dow'].apply(is_weekend)  # 是否周末
    
    # 删除临时日期时间特征
    del data["DT"]
    
    # 特征选择
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

