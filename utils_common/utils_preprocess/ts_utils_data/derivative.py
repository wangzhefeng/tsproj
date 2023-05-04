# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_processing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 序列数据预处理类函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import pysindy as ps


def GetSeriesDerivative(x_data, y_data, dev_method="Spline"):
    """
    依据给定方法求y_data序列对x_data的导数
    Parameters:
        x_data: 变量x序列
        y_data: 因变量序列
        dev_method: 微分方法
    Returns:
        微分结果
    """
    diffs = {
        'FiniteDifference': ps.FiniteDifference(),
        'Finite Difference': ps.SINDyDerivative(kind='finite_difference', k=1),
        'SmoothedFiniteDifference': ps.SmoothedFiniteDifference(),
        'SavitzkyGolay': ps.SINDyDerivative(kind='savitzky_golay', left=0.5, right=0.5, order=3),
        'Spline': ps.SINDyDerivative(kind='spline', s=1e-2),
        'TrendFiltered': ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-2),
        'Spectral': ps.SINDyDerivative(kind='spectral')
    }
    if diffs.get(dev_method) is not None:
        return diffs[dev_method](y_data, x_data)
    return None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

