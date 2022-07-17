# -*- coding: utf-8 -*-


# ***************************************************
# * File        : trend_remove.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None, "display.max_row", None)
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams["figure.figsize"] = 15, 6


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def trend_remove(ts: pd.DataFrame, 
                 method: str, 
                 chart: bool = False) -> None:
    """
    趋势性去除

    :param ts: _description_
    :type ts: pd.DataFrame
    :param method: _description_
    :type method: str
    :param chart: _description_, defaults to False
    :type chart: bool, optional
    :return: _description_
    :rtype: _type_
    """
    if method == "log":
        # 对数转换消除趋势
        ts_log = np.log(ts)
        if chart:
            plt.plot(ts_log)
            plt.show()
        return ts_log
    elif method == "moving":
        # 移动平均消除趋势
        ts_log = np.log(ts)
        moving_avg = ts_log.rolling(window = 12).mean()
        if chart:
            plt.plot(ts_log)
            plt.plot(moving_avg, color = "red")
            plt.show()
        return moving_avg




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

