# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-17
# * Version     : 1.0.041713
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
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.ts.augmentation import run_augmentation_single
from utils.ts.timefeatures import time_features
from utils.filter_str import filter_number
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Dataset_Train(Dataset):
    
    def __init__(self, 
                 args,
                 root_path, 
                 data_path,
                 flag='train', 
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='MS', 
                 target='OT',
                 time="time",
                 freq='15min',
                 timeenc=0,
                 seasonal_patterns=None,
                 scale=True,
                 inverse=False,
                 testing_step=1):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        self.flag = flag
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.time = time
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.testing_step = testing_step
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.info(f"Train data shape: {df_raw.shape}")
        # 缺失值处理
        df_raw.dropna(axis=1, how='any', inplace=True)
        logger.info(f"Train data shape after dropna: {df_raw.shape}")
        # 删除方差为 0 的特征
        df_raw = df_raw.loc[:, (df_raw != df_raw.loc[0]).any()]
        logger.info(f"Train data shape after drop 0 variance: {df_raw.shape}")
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.time)
        df_raw = df_raw[[self.time] + cols + [self.target]]
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        # 根据预测任务进行特征筛选
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # 数据分割比例
        num_train = int(len(df_data) * self.args.train_ratio)  # 0.7
        num_test = int(len(df_data) * self.args.test_ratio)    # 0.2
        num_vali = len(df_data) - num_train - num_test         # 0.1
        # 数据分割索引
        border1s = [0,         num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali,     len(df_data)]
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
        logger.info(f"Train step: {1}, Valid step: {1}, Test step: {self.testing_step}")
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}, data length: {border2-border1}")
        # 时间特征处理
        df_stamp = df_raw[[self.time]]
        df_stamp = df_stamp[border1:border2]
        df_stamp[self.time] = pd.to_datetime(df_stamp[self.time])
        if self.timeenc == 0:
            freq_num = filter_number(self.freq)[0]
            df_stamp['month'] = df_stamp[self.time].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp[self.time].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp[self.time].map(lambda x: x // freq_num)
            data_stamp = df_stamp.drop([self.time], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        logger.info(f"Train timestamp features shape: {data_stamp.shape}")
        # 数据切分
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # TODO 数据增强
        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
        #         self.data_x, self.data_y, self.args
        #     )
        # logger.info(f"debug::data_x: \n{self.data_x} \ndata_x shape: {self.data_x.shape}")
        # logger.info(f"debug::data_y: \n{self.data_y} \ndata_y shape: {self.data_y.shape}")
        # logger.info(f"debug::data_stamp: \n{self.data_stamp} \ndata_stamp shape: {self.data_stamp.shape}")

    def __getitem__(self, index):
        # data_x 索引
        if self.flag in ["train", "valid"]:
            s_begin = index
        elif self.flag == "test":
            if self.testing_step == 1:
                s_begin = index
            elif self.testing_step == self.pred_len:
                s_begin = index * self.pred_len
            else:
                s_begin = index * self.testing_step
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # 时间特征分割
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
         
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    
    def __init__(self, 
                 args,
                 root_path, 
                 data_path,
                 flag='pred', 
                 size=None,  # size: [seq_len, label_len, pred_len]
                 features='MS',
                 target='OT', 
                 time="time",
                 timeenc=0, 
                 freq='15min',
                 seasonal_patterns=None,
                 scale=True, 
                 inverse=False,
                 testing_step=None):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        self.flag = flag
        assert flag in ["pred"]
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.time = time
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.testing_step=testing_step
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.info(f"Train data shape: {df_raw.shape}")
        # 缺失值处理
        df_raw.dropna(axis=1, how='any', inplace=True)
        logger.info(f"Train data shape after dropna: {df_raw.shape}")
        # 删除方差为 0 的特征
        df_raw = df_raw.loc[:, (df_raw != df_raw.loc[0]).any()]
        logger.info(f"Train data shape after drop 0 variance: {df_raw.shape}")
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.time)
        df_raw = df_raw[[self.time] + cols + [self.target]]
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        # 预测特征变量数据
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # TODO 数据转换 v1
        # self.scaler = StandardScaler()
        # if self.scale:
        #     self.scaler.fit(df_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
        # TODO 数据转换 v2
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        if self.scale:
            if self.features == 'M' or self.features == 'S':
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                self.scaler.fit(df_data.values[:, :-1])
                self.y_scaler.fit(df_data.values[:, -1].reshape(-1, 1))
                data_x = self.scaler.transform(df_data.values[:, :-1])
                data_y = self.y_scaler.transform(df_data.values[:, -1].reshape(-1, 1))
                data = np.concatenate((data_x, data_y), axis = 1)
        logger.info(f"Train data shape after standardization: {data.shape}")
        # 数据窗口索引
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        logger.info(f"Forecast input data index: {border1}:{border2}, data length: {border2-border1}")
        # 时间戳特征处理
        # history date
        forecast_history_stamp = df_raw[[self.time]][border1:border2]
        forecast_history_stamp[self.time] = pd.to_datetime(forecast_history_stamp[self.time], format='mixed')
        forecast_history_stamp = forecast_history_stamp[self.time].values
        # future date
        forecast_future_stamp = pd.date_range(forecast_history_stamp[-1], periods=self.pred_len + 1, freq=self.freq)
        forecast_future_stamp = forecast_future_stamp[1:].values
        self.forecast_start_time = forecast_future_stamp[0]
        # history + future date
        df_stamp = pd.DataFrame({self.time: list(forecast_history_stamp) + list(forecast_future_stamp)})
        if self.timeenc == 0:
            freq_num = filter_number(self.freq)[0]
            df_stamp['month'] = df_stamp[self.time].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp[self.time].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // freq_num)
            data_stamp = df_stamp.drop([self.time], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        logger.info(f"Train and Forecast timestamp features shape: {data_stamp.shape}")
        # 数据切分
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # logger.info(f"debug::data_x: \n{self.data_x} \ndata_x shape: {self.data_x.shape}")
        # logger.info(f"debug::data_y: \n{self.data_y} \ndata_y shape: {self.data_y.shape}")
        # logger.info(f"debug::data_stamp: \n{self.data_stamp} \ndata_stamp shape: {self.data_stamp.shape}")
    
    def __getitem__(self, index):
        # data_x 索引
        s_begin = index
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len 
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:(r_begin+self.label_len)]
        else:
            seq_y = self.data_y[r_begin:(r_begin+self.label_len)]
        # 时间特征分割
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        # TODO 数据逆转换 v1
        # return self.scaler.inverse_transform(data)
        # TODO 数据逆转换 v2
        if self.features == 'M' or self.features == 'S':
            return self.scaler.inverse_transform(data)
        else:
            return self.y_scaler.inverse_transform(data)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
