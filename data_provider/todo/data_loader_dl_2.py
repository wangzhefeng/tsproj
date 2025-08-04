# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_dl_todo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-06
# * Version     : 1.0.040614
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Data_Loader:

    def __init__(self, filename: str, train_ratio: float, cols: List):
        """
        Args:
            filename (str): 时序数据文件路径
            train_ratio (float): 训练集、测试集分割比例
            cols (List): 时序数据的特征名称
        """
        self.filename = filename
        self.train_ratio = train_ratio
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
        """
        数据分割
        """
        i_split = int(len(df) * self.train_ratio)
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
    from utils.ts.tsproj_dl.config.gru import Config
    
    configs = Config()

    # 读取数据
    data = Data_Loader(
        filename = os.path.join('data', configs['data']['filename']),
        train_ratio = configs['data']['train_test_split'],
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

if __name__ == "__main__":
    main()
