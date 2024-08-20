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
                 split_ratio: float = 0.8,
                 batch_size: int = 32) -> None:
        self.data = data
        self.timestep = timestep
        self.feature_size = feature_size
        self.output_size = output_size
        self.target_index = target_index
        self.split_ratio = split_ratio
        self.batch_size = batch_size

    # TODO
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
        self.x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.output_size)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        return [train_data, train_loader, test_data, test_loader]

    #! 推荐
    def RecursiveMultiStep(self):
        """
        递归多步预测(单步滚动预测)
        例如：多变量：123456789 => 12345-67、23456-78、34567-89...
        例如：单变量：123456789 => 123-4、234-5、345-6...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(self.data) - self.timestep):
            if self.target_index is not None:
                dataX.append(self.data[index:(index + self.timestep)][:, self.target_index])  # 单变量特征
            else:
                dataX.append(self.data[index:(index + self.timestep)][:, :])  # 多变量特征
            dataY.append(self.data[index + self.timestep][self.target_index])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)  # (batch_size, timestep, feature_size)
        self.y_train = dataY[:train_size].reshape(-1, 1)  # (batch_size, num_target)
        self.x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, 1)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        return [train_data, train_loader, test_data, test_loader]

    # TODO
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
        self.x_train = dataX[:train_size, :].reshape(-1, self.timestep, self.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.timestep, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.output_size)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        return [train_data, train_loader, test_data, test_loader]

    def _dataset_dataloader(self):
        """
        # 创建 torch Dataset 和 DataLoader
        """
        # 将 numpy.ndarray 类型数据转换成 torch.Tensor 类型数据
        x_train_tensor = torch.from_numpy(self.x_train).to(torch.float32)
        y_train_tensor = torch.from_numpy(self.y_train).to(torch.float32)
        x_test_tensor = torch.from_numpy(self.x_test).to(torch.float32)
        y_test_tensor = torch.from_numpy(self.y_test).to(torch.float32)
        # data set
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)
        # data loader
        train_loader = DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(dataset = test_data, batch_size = self.batch_size, shuffle = False)
        return [x_train_tensor, y_train_tensor, train_data, train_loader, x_test_tensor, y_test_tensor, test_data, test_loader]




# 测试代码 main 函数
def main():
    import pandas as pd

    # ------------------------------
    # data
    # ------------------------------
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
    print(data)
    # print(data.values)
    print(data.shape)
    # ------------------------------
    # data split
    # ------------------------------
    data_split = Datasplitor(
        data = data.values, 
        timestep = 1,
        feature_size = 1, 
        output_size = 1, 
        target_index = 0,
        split_ratio = 0.8,
        batch_size = 32,
    )
    train_data, train_loader, \
    test_data, test_loader = data_split.RecursiveMultiStep()

if __name__ == "__main__":
    main()
