# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_splitor.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-20
# * Version     : 1.0.012021
# * Description : https://blog.csdn.net/java1314777/article/details/134407174
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

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from utils.time_col_tools import time_col_distinct
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Dataset_Train(Dataset):

    def __init__(self, 
                 args, 
                 root_path, 
                 data_path,
                 target,
                 freq,
                 features,
                 seq_len,
                 pred_len,
                 step_size: int = 1,
                 scale: bool = True,
                 flag: str = "train"):
        # command line args
        self.args = args
        # data info
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.freq = freq
        # task
        self.features = features
        # data type
        self.flag = flag
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        # data size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step_size = step_size
        # data trans
        self.scale = scale
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 读取数据文件
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=["date"])
        logger.info(f"Train data: \n{df_raw.head()} \nTrain data shape: {df_raw.shape}")
        logger.info(f"Train data NA check: \n{df_raw.isna().sum()}")
        # 删除重复时间戳
        df_raw = time_col_distinct(df_raw, time_col="date")
        logger.info(f"Train data shape after drop timestamp duplicate: {df_raw.shape}")
        # 根据时间戳补全数据
        df_complete = pd.DataFrame({"date": pd.date_range(df_raw["date"].min(), df_raw["date"].max(), freq=self.freq)})
        for col in df_raw.columns:
            if col != "date":
                df_complete[col] = df_complete["date"].map(df_raw.set_index("date")[col])
        df_raw = df_complete
        logger.info(f"Train data shape after date complete: {df_raw.shape}")
        # 缺失值处理
        df_raw.set_index("date", inplace=True)
        df_raw = df_raw.interpolate(method='linear', limit_direction='both')
        df_raw = df_raw.dropna(axis=0)
        df_raw.reset_index(inplace=True)
        logger.info(f"Train data shape after interpolate and dropna: {df_raw.shape}")
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        # 根据预测任务进行特征筛选
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # TODO self.input_size = df_data.shape[1]
        df_stamp = df_raw[['date']]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # 数据分割比例
        num_train = int(len(df_data) * self.args.train_ratio)  # 0.7
        num_test = int(len(df_data) * self.args.test_ratio)    # 0.2
        num_vali = len(df_data) - num_train - num_test         # 0.1
        # 数据分割索引
        border1s = [0,         num_train,          num_train]
        border2s = [num_train, num_train+num_vali, len(df_data)]
        # 数据分割
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        logger.info(f"Train data shape after standardization: {data.shape}")
        # 训练/测试/验证数据集分割: 选取当前 flag 下的数据
        logger.info(f"Train data length: {border2s[0]-border1s[0]}, Valid data length: {border2s[1]-border1s[1]}, Test data length: {border2s[2]-border1s[2]}")
        logger.info(f"Train step: {1}, Valid step: {1}, Test step: {1}")
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}, data length: {border2-border1}")
        # 数据切分
        data_tensor = torch.FloatTensor(data[border1:border2])
        # 创建 Dataset
        self.sequences = self.__create_input_sequences(data_tensor)
    
    def __create_input_sequences(self, input_data):
        """
        创建时间序列数据专用的数据分割器
        """
        output_seq = []
        input_data_len = len(input_data)
        for i in range(0, input_data_len - self.seq_len, self.step_size):
            train_seq = input_data[i:(i + self.seq_len)]
            if (i + self.seq_len + self.pred_len) > input_data_len:
                break 
            if self.features == "MS" or self.features == "S":
                train_label = input_data[:, -1:][(i + self.seq_len):(i + self.seq_len + self.pred_len)]
            else:
                train_label = input_data[(i + self.seq_len):(i + self.seq_len + self.pred_len)]
            output_seq.append((train_seq, train_label))
        # logger.info(f"output_seq[0][0]: \n{output_seq[0][0]} \noutput_seq[0][0].shape: {output_seq[0][0].shape}")
        # logger.info(f"output_seq[0][1]: \n{output_seq[0][1]} \noutput_seq[0][1].shape: {output_seq[0][1].shape}")
        # 样本数量
        sample_num = input_data_len - (self.seq_len + self.pred_len - 1)
        logger.info(f"{self.flag.capitalize()} sample number: {sample_num}")
        
        return output_seq
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
