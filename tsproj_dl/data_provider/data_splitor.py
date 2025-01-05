# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_splitor.py
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
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Data_Loader:

    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        self.data_path = self.cfgs.data_path
        self.target = self.cfgs.target
        self.seq_len = self.cfgs.seq_len
        self.feature_size = self.cfgs.feature_size
        self.output_size = self.cfgs.output_size
        self.target_index = self.cfgs.target_index
        self.features = self.cfgs.features
        self.pred_method = self.cfgs.pred_method
        self.split_ratio = self.cfgs.split_ratio
        self.batch_size = self.cfgs.batch_size
    
    def _read_data(self):
        """
        data read
        """
        if self.data_path is not None:
            data = pd.read_csv(self.data_path, index_col = 0)
        else:
            data = self.cfgs.data
        
        return data

    def _transform_data(self, df):
        """
        data scaler
        """
        # min-max scaler
        scaler_model = MinMaxScaler()
        data = scaler_model.fit_transform(np.array(df))
        # min-max scaler
        self.scaler = MinMaxScaler()
        self.scaler.fit_transform(np.array(df[self.target]).reshape(-1, 1))
        
        return data
    
    def RecursiveMultiStep(self, data):
        """
        递归多步预测(单步滚动预测)
        
        例如：多变量：123456789 => 12345-67、23456-78、34567-89...
        例如：单变量：123456789 => 123-4、234-5、345-6...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - self.seq_len):
            if self.features == "S":
                dataX.append(data[index:(index + self.seq_len)][:, self.target_index])  # 单变量特征
            else:
                dataX.append(data[index:(index + self.seq_len)][:, :])  # 多变量特征
            dataY.append(data[index + self.seq_len][self.target_index]) 
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # print(f"dataX: \n{dataX}")
        # print(f"dataX shape: {dataX.shape}")
        # print(f"dataY: \n{dataY}")
        # print(f"dataY shape: {dataY.shape}")
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.seq_len, self.feature_size)  # (batch_size, seq_len, feature_size)
        self.y_train = dataY[:train_size].reshape(-1, 1)  # (batch_size, num_target)
        self.x_test = dataX[train_size:, :].reshape(-1, self.seq_len, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, 1)
        # print(f"x_train: \n{self.x_train}")
        # print(f"x_train shape: {self.x_train.shape}")
        # print(f"y_train: \n{self.y_train}")
        # print(f"y_train shape: {self.y_train.shape}")
        # 创建 torch Dataset 和 DataLoader
        [train_loader, test_loader] = self._dataset_dataloader()
        
        return [train_loader, test_loader]
    
    def DirectMultiStepOutput(self, data):
        """
        直接多步预测
        
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - self.seq_len - 1):
            dataX.append(data[index:(index + self.seq_len)])
            dataY.append(data[(index + self.seq_len):(index + self.seq_len + self.output_size)][:, self.target_index].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.seq_len, self.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.seq_len, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.output_size)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        
        return [train_data, train_loader, test_data, test_loader] 

    def DirectRecursiveMix(self, data):
        """
        直接递归混合预测(多模型滚动预测)
        
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - self.seq_len - 1):
            dataX.append(data[index:(index + self.seq_len)][:, self.target_index])
            dataY.append(data[(index + self.seq_len):(index + self.seq_len + self.output_size)][:, self.target_index].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.seq_len, self.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.seq_len, self.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.output_size)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        
        return [train_data, train_loader, test_data, test_loader]

    def _dataset_dataloader(self):
        """
        # 创建 torch Dataset 和 DataLoader
        """
        # 将 numpy.ndarray 类型数据转换成 torch.Tensor 类型数据
        self.x_train_tensor = torch.from_numpy(self.x_train).to(torch.float32)
        self.y_train_tensor = torch.from_numpy(self.y_train).to(torch.float32)
        self.x_test_tensor = torch.from_numpy(self.x_test).to(torch.float32)
        self.y_test_tensor = torch.from_numpy(self.y_test).to(torch.float32)
        # data set
        train_data = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        test_data = TensorDataset(self.x_test_tensor, self.y_test_tensor)
        # data loader
        train_loader = DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(dataset = test_data, batch_size = self.batch_size, shuffle = False)
        
        return [train_loader, test_loader]

    def run(self):
        # 读取数据
        data = self._read_data()
        # 数据预处理
        data = self._transform_data(data)
        # 选择预测方法
        if self.pred_method == "recursive_multi_step":
            return self.RecursiveMultiStep(data)
        elif self.pred_method == "direct_multi_step_output":
            return self.DirectMultiStepOutput(data)
        elif self.pred_method == "direct_recursive_mix":
            return self.DirectRecursiveMix(data)




# 测试代码 main 函数
def main():
    from tsproj_csdn.config.gru import Config_test
    
    config = Config_test()
    # ------------------------------
    # data split
    # ------------------------------
    data_split = Data_Loader(cfgs=config)
    train_data, train_loader, test_data, test_loader = data_split.run()

if __name__ == "__main__":
    main()
