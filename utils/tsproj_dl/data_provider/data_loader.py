# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_splitor.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-25
# * Version     : 0.1.052522
# * Description : https://weibaohang.blog.csdn.net/article/details/128595011
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
        self.features = self.cfgs.features
        self.target = self.cfgs.target
        self.target_index = self.cfgs.target_index
        self.seq_len = self.cfgs.seq_len
        self.feature_size = self.cfgs.feature_size
        self.output_size = self.cfgs.output_size
        self.split_ratio = self.cfgs.split_ratio
        self.batch_size = self.cfgs.batch_size
        self.pred_method = self.cfgs.pred_method
        self.scale = self.cfgs.scale
    
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
        if self.scale:
            # min-max scaler
            scaler_model = MinMaxScaler()
            data = scaler_model.fit_transform(np.array(df))
            # min-max scaler
            self.scaler = MinMaxScaler()
            self.scaler.fit_transform(np.array(df[self.target]).reshape(-1, 1))
        else:
            data = np.array(df)
        
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
                dataX.append(data[index:(index + self.seq_len)][:, self.target_index:self.feature_size])  # 多变量特征
            dataY.append(data[index + self.seq_len][self.target_index])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # print(f"dataX: \n{dataX}")
        # print(f"dataX shape: {dataX.shape}")
        # print(f"dataY: \n{dataY}")
        # print(f"dataY shape: {dataY.shape}")
        print("-" * 80)
        # 训练集大
        train_size = int(np.round(self.split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.seq_len, self.feature_size)  # (batch_size, seq_len, feature_size)
        self.y_train = dataY[:train_size].reshape(-1, 1)  # (batch_size, num_target)
        self.x_test = dataX[train_size:, :].reshape(-1, self.seq_len, self.feature_size)  # (batch_size, seq_len, feature_size)
        self.y_test = dataY[train_size:].reshape(-1, 1)  # (batch_size, num_target)
        # print(f"x_train: \n{self.x_train}")
        # print(f"x_train shape: {self.x_train.shape}")
        # print(f"y_train: \n{self.y_train}")
        # print(f"y_train shape: {self.y_train.shape}")
        # print("-" * 40)
        # print(f"x_test: \n{self.x_test}")
        # print(f"x_test shape: {self.x_test.shape}")
        # print(f"y_test: \n{self.y_test}")
        # print(f"y_test shape: {self.y_test.shape}")
        # 创建 torch Dataset 和 DataLoader
        train_loader, test_loader = self._dataset_dataloader()
        
        return train_loader, test_loader
    
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
        
        return train_loader, test_loader

    def run(self):
        # 读取数据
        data = self._read_data()
        data = data.head(10)
        print(data)
        print("-" * 80)
        # 数据预处理
        data = self._transform_data(data)
        print(data)
        print("-" * 80)
        # 选择预测方法
        if self.pred_method == "recursive_multi_step":
            return self.RecursiveMultiStep(data)
        elif self.pred_method == "direct_multi_step_output":
            return self.DirectMultiStepOutput(data)
        elif self.pred_method == "direct_recursive_mix":
            return self.DirectRecursiveMix(data)


class Data_Loader_todo:

    def __init__(self, filename: str, split_ratio: float, cols: List):
        """
        Args:
            filename (str): 时序数据文件路径
            split_ratio (float): 训练集、测试集分割比例
            cols (List): 时序数据的特征名称
        """
        self.filename = filename
        self.split_ratio = split_ratio
        self.cols = cols
        self.len_train_windows = None
        # 数据读取
        df = self._read_data()
        # 数据划分
        self._split_data(df)

    def _read_data(self):
        """
        数据读取
        """
        df = pd.read_csv(self.filename)
        
        return df

    def _split_data(self, df):
        i_split = int(len(df) * self.split_ratio)
        self.data_train = df.get(self.cols).values[:i_split]
        self.data_test  = df.get(self.cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)

    def get_train_data(self, seq_len: int, normalise: bool):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough 
        memory to load data, otherwise use generate_train_batch() method.

        Args:
            seq_len (int): 划窗序列窗口长度
            normalise (bool): 是否进行归一化

        Returns:
            _type_: 训练数据 x, y
        """
        # sliding windows
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            # 生成下一个窗口数据
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        
        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len: int, normalise: bool):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.

        Args:
            seq_len (int): 划窗序列窗口长度
            normalise (bool): 是否进行归一化

        Returns:
            _type_: 测试数据 x, y
        """
        # sliding windows
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        # 归一化
        data_windows = self.normalise_windows(
            window_data = data_windows, 
            single_window = False
        ) if normalise else data_windows
        # 数据分割
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        
        return x, y

    def generate_train_batch(self, seq_len: int, batch_size: int, normalise: bool):
        """
        Yield a generator of training data from filename 
        on given list of cols split for train/test

        Args:
            seq_len (int): 划窗序列窗口长度
            batch_size (int): batch size
            normalise (bool): 是否进行归一化

        Yields:
            _type_: 训练数据 x_batch, y_batch
        """
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                # stop-condition for a smaller final batch if data doesn't divide evenly
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                # 生成下一个窗口数据
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)

                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i: int, seq_len: int, normalise: bool):
        """
        Generates the next data window from the given index location i

        Args:
            i (int): 划窗的开始数据点索引
            seq_len (int): 划窗序列窗口长度
            normalise (bool): 是否进行归一化

        Returns:
            _type_: 单个窗口的 x, y
        """
        # sliding windows
        window = self.data_train[i:i+seq_len]
        # normalize
        window = self.normalise_windows(
            window_data = window, 
            single_window = True
        )[0] if normalise else window
        # window split
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data: np.array, single_window: bool = False):
        """
        Normalise window with a base value of zero

        Args:
            window_data (np.array): 窗口数据
            single_window (bool, optional): 是否是单个窗口的数据. Defaults to False.

        Returns:
            _type_: 归一化后的时序数组
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [float(p) / float(window[0, col_i]) - 1 for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        
        return np.array(normalised_data)




# 测试代码 main 函数
def main():
    from utils.tsproj_dl.config.gru import Config
    
    config = Config()
    # ------------------------------
    # data split
    # ------------------------------
    data_loader = Data_Loader(cfgs = config)
    train_loader, test_loader = data_loader.run()
    '''
    # ------------------------------
    # 读取数据
    # ------------------------------
    data = Data_Loader_todo(
        filename = os.path.join('data', configs['data']['filename']),
        split_ratio = configs['data']['train_test_split'],
        cols = configs['data']['columns'],
    )
    # 训练数据
    x, y = data.get_train_data(
        seq_len = configs['data']['sequence_length'],
        normalise = configs['data']['normalise']
    )
    logger.info(f"x shape={x.shape}")
    logger.info(f"y shape={y.shape}")
    # 测试数据
    x_test, y_test = data.get_test_data(
        seq_len = configs['data']['sequence_length'],
        normalise = configs['data']['normalise']
    )
    logger.info(f"x_test shape={x_test.shape}")
    logger.info(f"y_test shape={y_test.shape}")
    '''

if __name__ == "__main__":
    main()
