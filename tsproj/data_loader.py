# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_load.py
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

import pandas as pd


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def get_ts_df(filepath: str, ts_col: str, ts_format: str = "%Y-%m-%d", log: str = False) -> pd.DataFrame:
    """
    _summary_

    :param filepath: _description_
    :type filepath: str
    :param ts_col: _description_
    :type ts_col: str
    :param ts_format: _description_, defaults to "%Y-%m-%d"
    :type ts_format: str, optional
    :param log: _description_, defaults to False
    :type log: str, optional
    :return: _description_
    :rtype: pd.DataFrame
    """
    dateparse = lambda dates: pd.to_datetime(dates, format = ts_format)
    ts_df = pd.read_csv(
        filepath,
        parse_dates = [ts_col],
        index_col = ts_col, 
        date_parser = dateparse,
    )
    if log:
        print(ts_df.shape)
        print("-" * 25)
        print(ts_df.head())
        print("-" * 25)
        print(ts_df.dtypes)
        print("-" * 25)
        print(ts_df.index)
        print("-" * 25)




__all__ = [
    get_ts_df,

]


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

