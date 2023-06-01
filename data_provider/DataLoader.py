# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052700
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

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_provider.data_generator import split_data

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DataLoader:
    """
    A class for loading and transforming data for the lstm model
    """

    def __init__(self, filename: str, split_ratio: float, cols: List):
        """
        Args:
            filename (str): 时序数据文件路径
            split_ratio (float): 训练集、测试集分割比例
            cols (List): 时序数据的特征名称
        """
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split_ratio)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

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
        data_windows = self.normalise_windows(window_data = data_windows, single_window = False) if normalise else data_windows
        # 数据分割
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

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
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            # 生成下一个窗口数据
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

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
                # ???stop-condition for a smaller final batch if data doesn't divide evenly
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
        window = self.normalise_windows(window_data = window, single_window = True)[0] if normalise else window
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



class Config:
    data_path = "data/wind_dataset.csv"
    timestep = 1  # 时间步长，就是利用多少时间窗口 #TODO window_len
    feature_size = 1  # 每个步长对应的特征数量，这里只使用 1 维(每天的风速)
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 0.0003  # 学习率
    best_loss = 0  # 记录损失
    split_ratio = 0.8  # 训练测试数据分割比例
    model_name = "lstm"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


class DataLoaderV2:
    
    def __init__(self, config) -> None:
        self.config = config
    
    def __init__(self, config,  cols: List):
        """
        Args:
            filename (str): 时序数据文件路径
            split_ratio (float): 训练集、测试集分割比例
            cols (List): 时序数据的特征名称
        """
        dataframe = pd.read_csv(config.data_path, index_col = 0)
        i_split = int(len(dataframe) * config.split_ratio)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def minmaxscale_windows(self):
        # data scaler
        wind_array = np.array(self.dataframe["WIND"]).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit_transform(wind_array)

        scaler_model = MinMaxScaler()
        data = scaler_model.fit_transform(np.array(self.dataframe))

    def build_dataloader(self, data, config):
        # data split
        x_train, y_train, x_test, y_test = split_data(
            data = data, 
            timestep = config.timestep, 
            input_size = config.feature_size,
            split_ratio = 0.8,
        )
        logger.info(f"\nx_train: \n{x_train}")
        logger.info(f"\ny_train: \n{y_train}")

        # data loader
        train_data = TensorDataset(
            torch.from_numpy(x_train).to(torch.float32), 
            torch.from_numpy(y_train).to(torch.float32)
        )
        test_data = TensorDataset(
            torch.from_numpy(x_test).to(torch.float32), 
            torch.from_numpy(y_test).to(torch.float32)
        )
        train_loader = DataLoader(dataset = train_data, batch_size = config.batch_size, shuffle = False)
        test_loader = DataLoader(dataset = test_data, batch_size = config.batch_size, shuffle = False)
  
    def split_data(self, data, timestep: int, input_size: int, split_ratio: float = 0.8):
        """
        形成训练数据
        例如：123456789 => 12-3、23-4、34-5...

        Args:
            data (_type_): _description_
            timestep (int): _description_
            input_size (int): _description_
            split_ratio (float, optional): _description_. Defaults to 0.8.

        Returns:
            _type_: _description_
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - timestep):
            dataX.append(data[index:index + timestep][:, 0])
            dataY.append(data[index + timestep][0])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(split_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        logger.info(dataX)
        logger.info(dataX[:2, :])
        x_train = dataX[:train_size, :].reshape(-1, timestep, input_size)
        y_train = dataY[:train_size].reshape(-1, 1)

        x_test = dataX[train_size:, :].reshape(-1, timestep, input_size)
        y_test = dataY[train_size:].reshape(-1, 1)

        return [x_train, y_train, x_test, y_test]

    # TODO
    def test(self):
        train_preds = self.scaler.inverse_transform(train_preds)
        train_true = self.scaler.inverse_transform(train_true)
        test_preds = self.scaler.inverse_transform(test_preds)
        test_true = self.scaler.inverse_transform(test_true)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
