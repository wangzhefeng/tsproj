# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_preprocess.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-13
# * Version     : 0.1.051316
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

from loguru import logger
from pandas.tseries import to_offset

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_preprocess(df):
    """
    数据预处理
    1. 数据排序
    2. 去除重复值
    3. 重采样（ 可选）
    4. 缺失值处理
    5. 异常值处理
    """
    # 排序
    df = df.sort_values(by = "DATATIME", ascending = True)
    logger.info(f"df.shape: {df.shape}")
    logger.info(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")
    # 去除重复值
    df = df.drop_duplicates(subset = "DATATIME", keep = "first")
    logger.info(f"After dropping dulicates: {df.shape}")
    # 重采样（可选）+ 缺失值处(理线性插值)：比如 04 风机缺少 2022-04-10 和 2022-07-25 两天的数据，重采样会把这两天数据补充进来
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.set_index("DATATIME")
    df = df.resample(rule = to_offset('15T').freqstr, label = 'right', closed = 'right')
    df = df.interpolate(method = 'linear', limit_direction = 'both').reset_index()
    # 异常值处理
    # 当实际风速为 0 时，功率设置为 0
    df.loc[df["ROUND(A.WS,1)"] == 0, "YD15"] = 0
    # TODO 风速过大但功率为 0 的异常：先设计函数拟合出：实际功率=f(风速)，然后代入异常功率的风速获取理想功率，替换原异常功率
    # TODO 对于在特定风速下的离群功率（同时刻用 IQR 检测出来），做功率修正（如均值修正）

    return df




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
