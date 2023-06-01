# -*- coding: utf-8 -*-

# ***************************************************
# * File        : split_data.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-25
# * Version     : 0.1.052522
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
import numpy as np

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def split_data(data, timestep: int, feature_size: int, split_ratio: float = 0.8):
    """
    !Informer
    单变量、多变量-单步、多步预测
    例如：123456789 => 12345-67、23456-78、34567-89...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep):
        dataX.append(data[index:(index+timestep)][:, :])
        dataY.append(data[index+timestep][0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]


def split_data(data, timestep: int, feature_size: int, output_size, split_ratio: float = 0.8):
    """
    !直接多输出预测
    单变量、多变量-单步、多步预测
    例如：123456789 => 12345-67、23456-78、34567-89...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep - 1):
        dataX.append(data[index:(index+timestep)])
        dataY.append(data[(index+timestep):(index+timestep+output_size)][:, 0].tolist())
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, output_size)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, output_size)

    return [x_train, y_train, x_test, y_test]


def split_data(data, timestep: int, feature_size: int, split_ratio: float = 0.8):
    """
    !递归多步预测(单步滚动预测)
    单变量、多变量-单步、多步预测
    例如：123456789 => 123-4、234-5、345-6...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep):
        dataX.append(data[index:(index+timestep)][:, 0])
        dataY.append(data[index+timestep][0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]


def split_data(data, timestep: int, feature_size: int, output_size: int, split_ratio: float = 0.8):
    """
    !直接多步预测(多模型单步预测)
    单变量、多变量-单步、多步预测
    例如：123456789 => 12345-67、23456-78、34567-89...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep - 1):
        dataX.append(data[index:(index+timestep)])
        dataY.append(data[(index+timestep):(index+timestep+output_size)][:, 0].tolist())
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, output_size)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, output_size)

    return [x_train, y_train, x_test, y_test]


def split_data(data, timestep: int, feature_size: int, output_size: int, split_ratio: float = 0.8):
    """
    !直接递归混合预测(多模型滚动预测)
    单变量、多变量-单步、多步预测
    例如：123456789 => 12345-67、23456-78、34567-89...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep - 1):
        dataX.append(data[index:(index+timestep)][:, 0])
        dataY.append(data[(index+timestep):(index+timestep+output_size)][:, 0].tolist())
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, output_size)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, output_size)

    return [x_train, y_train, x_test, y_test]


def split_data(data, timestep: int, feature_size: int, output_size: int, split_ratio: float = 0.8):
    """
    !Seq2Seq 多步预测
    单变量、多变量-单步、多步预测
    例如：123456789 => 12345-67、23456-78、34567-89...
    """
    dataX = []  # 保存 X
    dataY = []  # 保存 Y
    # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
    for index in range(len(data) - timestep - 1):
        dataX.append(data[index:(index+timestep)])
        dataY.append(data[(index+timestep):(index+timestep+output_size)][:, 0].tolist())
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # 训练集大
    train_size = int(np.round(split_ratio * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[:train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[:train_size].reshape(-1, output_size)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, output_size)

    return [x_train, y_train, x_test, y_test]





# 测试代码 main 函数
def main():
    import pandas as pd

    # TODO
    data = pd.DataFrame({
        "Date": pd.to_datetime(["1961-01-01", "1961-01-02", "1961-01-03", "1961-01-04", "1961-01-05"]),
        "Wind": [13.67, 11.50, 11.25, 8.63, 11.92],
        "Temperature": [12, 18, 13, 27, 5],
        "Rain": [134, 234, 157, 192, 260],
    })
    data.set_index("Date", inplace = True)
    logger.info(data)
    logger.info(data.values)

    x_train, y_train, x_test, y_test = split_data(
        data = data.values, 
        timestep = 1,
        input_size = 2, 
        split_ratio = 0.8
    )
    logger.info(x_train)
    logger.info(y_train)
    logger.info(x_test)
    logger.info(y_test)

if __name__ == "__main__":
    main()
