# -*- coding: utf-8 -*-

# ***************************************************
# * File        : heteroskedasticity_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-08
# * Version     : 0.1.090822
# * Description : 使用统计检验来检查时间序列是否为异方差序列主要有三种方法：
# *               - White Test
# *               - Breusch-Pagan Test
# *               - Goldfeld-Quandt Test
# *               这些检验的主要输入是回归模型的残差(如普通最小二乘法)。
# *               零假设是残差的分布方差相等。如果 p 值小于显著性水平，则拒绝该假设。
# *               这就说明时间序列是异方差的， 检验显著性水平通常设置为 0.05。
# * Link        : https://github.com/vcerqueira/blog/blob/main/src/heteroskedasticity.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from typing import Dict
import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


TEST_NAMES = ["White", "Breusch-Pagan", "Goldfeld-Quandt"]
FORMULA = 'value ~ time'


class Heteroskedasticity:
 
    @staticmethod
    def het_tests(series: pd.Series, test: str) -> float:
        """
        Testing for heteroskedasticity
        异方差检验

        Parameters:
            series: Univariate time series as pd.Series
            test: String denoting the test. One of 'White','Goldfeld-Quandt', or 'Breusch-Pagan'
        Return:
            p-value as a float.
 
        If the p-value is high, we accept the null hypothesis that the data is homoskedastic
        """
        # 测试方法验证
        assert test in TEST_NAMES, 'Unknown test'
        # 时间序列预处理
        series = series.reset_index(drop = True).reset_index()
        series.columns = ['time', 'value']
        series['time'] += 1
        # 最小二乘回归
        olsr = ols(FORMULA, series).fit()
        # 假设检验
        if test == 'White':
            _, p_value, _, _ = sms.het_white(
                olsr.resid, 
                olsr.model.exog
            )
        elif test == 'Goldfeld-Quandt':
            _, p_value, _ = sms.het_goldfeldquandt(
                olsr.resid, 
                olsr.model.exog, 
                alternative = 'two-sided'
            )
        else:
            _, p_value, _, _ = sms.het_breuschpagan(
                olsr.resid, 
                olsr.model.exog
            )
 
        return p_value

    @classmethod
    def run_all_tests(cls, series: pd.Series) -> Dict[str, float]:
        """
        运行建设检验

        Args:
            series (pd.Series): _description_

        Returns:
            Dict: 检验结果, p-value
        """
        test_results = {
            k: cls.het_tests(series, k) 
            for k in TEST_NAMES
        }
 
        return test_results


# 测试代码 main 函数
def main():
    from pmdarima.datasets import load_airpassengers
    # from heteroskedasticity import Heteroskedasticity
    series = load_airpassengers(True)
    test_results = Heteroskedasticity.run_all_tests(series)
    print(test_results)

if __name__ == "__main__":
    main()
