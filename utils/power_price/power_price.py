# -*- coding: utf-8 -*-

# ***************************************************
# * File        : power_price.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-03
# * Version     : 1.0.060317
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
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger

def conditions_price(x):
    if x.month == 7 and x.hour in [12, 13]:
        return 1.4021
    elif x.month == 7 and x.hour in [8, 9, 10, 11, 14, 18, 19, 20]:
        return 1.1369
    elif x.month == 7 and x.hour in [6, 7, 15, 16, 17, 21]:
        return 0.6654
    elif x.month == 7 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return 0.3117
    elif x.month == 8 and x.hour in [12, 13]:
        return 1.4145
    elif x.month == 8 and x.hour in [8, 9, 10, 11, 14, 18, 19, 20]:
        return 1.1464
    elif x.month == 8 and x.hour in [6, 7, 15, 16, 17, 21]:
        return 0.6698
    elif x.month == 8 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return 0.3123
    # month 9
    elif x.month == 9 and x.day in [1, 7, 8, 15, 16, 17, 21, 22, 28]  and x.hour in [0, 1, 2, 3, 4, 5, 22, 23]:
        return 0.1720
    elif x.month == 9 and x.hour in [8, 9, 10, 11, 12, 13, 14, 18, 19, 20]:
        return 1.0987
    elif x.month == 9 and x.hour in [6, 7, 15, 16, 17, 21]:
        return 0.6354
    elif x.month == 9 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return 0.2879


def conditions_category(x):
    if x.month == 7 and x.hour in [12, 13]:
        return "尖峰"
    elif x.month == 7 and x.hour in [8, 9, 10, 11, 14, 18, 19, 20]:
        return "峰"
    elif x.month == 7 and x.hour in [6, 7, 15, 16, 17, 21]:
        return "平"
    elif x.month == 7 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return "谷"
    elif x.month == 8 and x.hour in [12, 13]:
        return "尖峰"
    elif x.month == 8 and x.hour in [8, 9, 10, 11, 14, 18, 19, 20]:
        return "峰"
    elif x.month == 8 and x.hour in [6, 7, 15, 16, 17, 21]:
        return "平"
    elif x.month == 8 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return "谷"
    elif x.month == 9 and x.day in [1, 7, 8, 15, 16, 17, 21, 22, 28]  and x.hour in [0, 1, 2, 3, 4, 5, 22, 23]:
        return "深谷"
    elif x.month == 9 and x.hour in [8, 9, 10, 11, 12, 13, 14, 18, 19, 20]:
        return "峰"
    elif x.month == 9 and x.hour in [6, 7, 15, 16, 17, 21]:
        return "平"
    elif x.month == 9 and x.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        return "谷"


df = pd.DataFrame({"time": pd.date_range(start="2024-07-01", end="2024-09-30 23:59:59", freq="5min")})
df["value"] = df["time"].apply(conditions_price)
df["type"] = df["time"].apply(conditions_category)
df.to_csv("ele_price-2024-789.csv", encoding="utf_8_sig", index=False)
logger.info(f"df: \n{df}")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
