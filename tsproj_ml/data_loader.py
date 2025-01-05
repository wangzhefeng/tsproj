# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_ts.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101715
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import Union

from datetime import datetime
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def read_ts(filepath: str,
            date_col: Union[str, int],
            date_parser: str = None,
            date_format: str = None,
            index_col: str = None,
            log: bool = False) -> pd.DataFrame:
    """
    读取时间序列数据
    """
    if date_parser is not None:
        parser = date_parser
    elif date_parser is None:
        if date_format is not None:
            parser = lambda dates: datetime.strptime(dates, date_format)
        elif date_format is None:
            parser = None
    
    ts_df = pd.read_csv(
        filepath,
        header = 0,
        index_col = index_col,
        parse_dates = [date_col],
        date_parser = parser,
        squeeze = True,
    )
    if log:
        print("-" * 25)
        print(f"SHAPE: {ts_df.shape}")
        print("-" * 25)
        print(f"HEAD: \n{ts_df.head()}")
        print("-" * 25)
        print(f"DTYPES: {ts_df.dtypes}")
        print("-" * 25)
        print(f"INDEX: {ts_df.index}")
        print("-" * 25)
    
    return ts_df


# TODO
class TimeseriesData:
    pass


# TODO
class TimeseriesDataLoader:
    pass


# 测试代码 main 函数
def main():
    series = read_ts(
        filepath = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
        date_col = "Month",
        date_parser = lambda dates: datetime.strptime("190" + dates, "%Y-%m"),
    )
    print(series)

if __name__ == "__main__":
    main()
