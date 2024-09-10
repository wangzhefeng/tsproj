# -*- coding: utf-8 -*-

# ***************************************************
# * File        : seasonal_test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-11
# * Version     : 1.0.091100
# * Description : SARIMA 季节性差分自回归移动平均模型的季节性成分参数估计
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pmdarima.arima.utils import nsdiffs

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def seasonal_D_estimate(series, seasonal_periods: int = 10, method = "CH"):
    """
    使用检验估计季节差分项(D)
    """
    # estimate number of seasonal differences using a Canova-Hansen test
    if method == "CH":
        D = nsdiffs(
            series, 
            m = seasonal_periods,  # commonly requires knowledge of dataset
            max_D = 12,
            test = 'ch'
        )  # -> 0
    else:
        # or use the OCSB test (by default)
        D = nsdiffs(
            series,
            m = seasonal_periods,
            max_D = 12,
            test = 'ocsb'
        )  # -> 0
    
    return D



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
