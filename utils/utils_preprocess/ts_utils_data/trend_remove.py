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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

rcParams["figure.figsize"] = 15, 6

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def trend_remove(ts: pd.DataFrame, method: str, chart: bool = False) -> None:
    """
    趋势性去除
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
