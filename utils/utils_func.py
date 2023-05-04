# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_func.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-18
# * Version     : 0.1.071822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "is_weekend",
]

# python libraries
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def is_weekend(row: int) -> int:
    """
    判断是否是周末
    
    Args:
        row (int): 一周的第几天

    Returns:
        int: 0: 不是周末, 1: 是周末
    """
    if row == 5 or row == 6:
        return 1
    else:
        return 0

def season(month: int) -> str:
    """
    判断当前月份的季节
    Args:
        day (_type_): _description_

    Returns:
        str: _description_
    """
    pass


def business_season(month: int) -> str:
    """
    业务季度

    Args:
        month (int): _description_

    Returns:
        str: _description_
    """
    pass


def workday_nums(month):
    """
    每个月中的工作日天数

    Args:
        month (_type_): _description_
    """
    pass


def holiday_nums(month):
    """
    每个月中的休假天数

    Args:
        month (_type_): _description_
    """
    pass


def is_summary_time(month):
    """
    是否夏时制

    Args:
        month (_type_): _description_
    """
    pass


def week_of_month(day):
    """
    一月中的第几周

    Args:
        day (_type_): _description_
    """
    pass


def is_holiday(day):
    pass


def holiday_continue_days(holiday):
    """
    节假日连续天数

    Args:
        holiday (_type_): _description_
    """
    pass


def holiday_prev_day_nums(holiday):
    """
    节假日前第 n 天

    Args:
        holiday (_type_): _description_
    """
    pass


def holiday_day_idx(holiday, day):
    """
    节假日第 n 天

    Args:
        holiday (_type_): _description_
        day (_type_): _description_
    """
    pass


def is_tiaoxiu(day):
    """
    是否调休

    Args:
        day (_type_): _description_
    """
    pass


def past_minutes(datetimes):
    """
    一天过去了几分钟

    Args:
        datetimes (_type_): _description_
    """
    pass


def time_period(date):
    """
    一天的哪个时间段

    Args:
        date (_type_): _description_
    """
    pass

def is_work(time):
    """
    该时间点是否营业/上班

    Args:
        time (_type_): _description_
    """
    pass





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
