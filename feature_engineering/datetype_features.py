# -*- coding: utf-8 -*-

# ***************************************************
# * File        : datetype_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-05
# * Version     : 1.0.040517
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def extend_datetype_features(df: pd.DataFrame, df_date: pd.DataFrame=None):
    """
    增加日期类型特征：
    1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
    """ 
    if df_date is None:
        return df, []
    else:
        # history data copy
        df_history_with_datetype = df.copy()
        df_date_history_copy = df_date.copy()
        # 构造日期特征
        df_history_with_datetype["date"] = df_history_with_datetype["ds"].apply(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        # 匹配日期类型特征
        df_history_with_datetype["date_type"] = df_history_with_datetype["date"].map(
            df_date_history_copy.set_index("date")["date_type"]
        )
        # 日期类型特征
        date_features = ["date_type"]
        return df_history_with_datetype, date_features




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
