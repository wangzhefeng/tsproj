# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ols_resid.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-09
# * Version     : 0.1.090900
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from typing import List
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_resid_exog(series: pd.Series, cols: List[str] = [], formula: str = ""):
    """
    最小二乘法拟合时序数据，获取数据残差和 TODO

    Args:
        series (pd.Series): _description_
        cols (List[str], optional): _description_. Defaults to [].
        formula (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    # 时间序列预处理
    series = series.reset_index(drop = True).reset_index()
    series.columns = cols
    # 回归模型最小二乘拟合
    ols_r = smf.ols(formula, data = series).fit()
    # 模型拟合概览
    # print(ols_r.summary())
    # 残差误差项
    resid = ols_r.resid
    # TODO
    exog = ols_r.model.exog

    return resid, exog



# 测试代码 main 函数
def main():
    from pmdarima.datasets import load_airpassengers

    # data
    series = load_airpassengers(True)
    # 时间序列预处理
    series = series.reset_index(drop = True).reset_index()
    series.columns = ['time', 'value']
    series['time'] += 1
    
    # regression model
    FORMULA = 'value ~ time'
    ols_r = smf.ols(FORMULA, data = series).fit()
    # 模型拟合概览
    # print(ols_r.summary())
    # 残差误差项
    print(ols_r.resid)
    # TODO
    # print(ols_r.model.exog)

if __name__ == "__main__":
    main()
