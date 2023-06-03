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

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Datasplitor:

    def __init__(self, 
                 data, 
                 timestep: int, 
                 feature_size: int, 
                 output_size: int = None, 
                 target_index: int = 0,
                 split_ratio: float = 0.8) -> None:
        self.data = data
        self.timestep = timestep
        self.feature_size = feature_size
        self.output_size = output_size
        self.target_index = target_index
        self.split_ratio = split_ratio

    # !Informer
    def split_data(self):
        """
        # TODO
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(self.data) - self.timestep):
            dataX.append(self.data[index:(index + self.timestep)])
            dataY.append(self.data[index + self.timestep][self.target_index])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)
        y_train = dataY[:train_size].reshape(-1, 1)

        x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        y_test = dataY[train_size:].reshape(-1, 1)

        return [x_train, y_train, x_test, y_test]

    def DirectMultiStepOutput(self):
        """
        直接多输出预测
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(self.data) - self.timestep - 1):
            dataX.append(self.data[index:(index + self.timestep)])
            dataY.append(self.data[(index + self.timestep):(index + self.timestep + self.output_size)][:, self.target_index].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)
        y_train = dataY[:train_size].reshape(-1, self.output_size)

        x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        y_test = dataY[train_size:].reshape(-1, self.output_size)

        return [x_train, y_train, x_test, y_test]

    def RecursiveMultiStep(self):
        """
        递归多步预测(单步滚动预测)
        例如：123456789 => 123-4、234-5、345-6...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(self.data) - self.timestep):
            dataX.append(self.data[index:(index + self.timestep)][:, self.target_index])
            dataY.append(self.data[index + self.timestep][self.target_index])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)  # (batch_size, timestep, feature_size)
        y_train = dataY[:train_size].reshape(-1, 1)  # (batch_size, num_target)

        x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        y_test = dataY[train_size:].reshape(-1, 1)

        return [x_train, y_train, x_test, y_test]

    def DirectRecursiveMix(self):
        """
        直接递归混合预测(多模型滚动预测)
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(self.data) - self.timestep - 1):
            dataX.append(self.data[index:(index + self.timestep)][:, self.target_index])
            dataY.append(self.data[(index + self.timestep):(index + self.timestep + self.output_size)][:, self.target_index].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)
        y_train = dataY[:train_size].reshape(-1, self.output_size)

        x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        y_test = dataY[train_size:].reshape(-1, self.output_size)

        return [x_train, y_train, x_test, y_test]

    @staticmethod
    def numpy2tensor(x_train, y_train, x_test, y_test):
        """
        将 numpy.ndarray 类型数据转换成 torch.Tensor 类型数据
        """
        x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
        return [x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor]

    @staticmethod
    def dataset_dataloader(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, batch_size):
        """
        创建 torch Dataset 和 DataLoader
        """
        # data set
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)
        # data loader
        train_loader = DataLoader(train_data, batch_size, False)
        test_loader = DataLoader(test_data, batch_size, False)
        return train_data, train_loader, test_data, test_loader




# 测试代码 main 函数
def main():
    import pandas as pd

    # data
    data = pd.DataFrame({
        "Date": pd.to_datetime([
            "1961-01-01", "1961-01-02", "1961-01-03", "1961-01-04", "1961-01-05",
            "1961-01-06", "1961-01-07", "1961-01-08", "1961-01-09", "1961-01-10",
        ]),
        "Wind": [13.67, 11.50, 11.25, 8.63, 11.92, 12.3, 11.5, 13.22, 11.77, 10.51],
        "Temperature": [12, 18, 13, 27, 5, 12, 15, 20, 22, 13],
        "Rain": [134, 234, 157, 192, 260, 167, 281, 120, 111, 223],
    })
    data.set_index("Date", inplace = True)
    # print(data)
    # print(data.values)
    # print(data.shape)
    
    data_split = Datasplitor(
        data = data.values, 
        timestep = 2,
        feature_size = 1, 
        output_size = 1, 
        target_index = 0,
        split_ratio = 0.8
    )
    x_train, y_train, \
    x_test, y_test = data_split.RecursiveMultiStep()
    print(f"x_train: {x_train}, x_train.shape: {x_train.shape}")
    print(f"y_train: {y_train}, y_train.shape: {y_train.shape}")
    print(f"x_test: {x_test}, x_test.shape: {x_test.shape}")
    print(f"y_test: {y_test}, y_test.shape: {y_test.shape}")
    x_train_tensor, y_train_tensor, \
    x_test_tensor, y_test_tensor = data_split.numpy2tensor(
        x_train, y_train, 
        x_test, y_test
    )
    train_data, train_loader, \
    test_data, test_loader = data_split.dataset_dataloader(
        x_train_tensor, y_train_tensor, 
        x_test_tensor, y_test_tensor, 
        batch_size = 32
    )

if __name__ == "__main__":
    main()
