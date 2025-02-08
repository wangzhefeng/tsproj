# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-14
# * Version     : 1.0.011420
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features
from utils.log_util import logger

warnings.filterwarnings('ignore')

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Dataset_Custom(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path,
                 pre_data=None,
                 flag='train', 
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='S', 
                 target='OT',
                 scale=True,
                 timeenc=0,
                 freq='h',
                 inverse = True,
                 cols = None):
        self.root_path = root_path
        self.data_path = data_path
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        self.features = features
        self.target = target
        self.freq = freq
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        # 读取数据
        self.__read_data__()

    def __read_data__(self):
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.dropna(axis=1, how='any', inplace=True)
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 预测特征变量数据
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # 训练/测试/验证数据集分割
        # 数据分割比例
        true_train = int(len(df_raw))                  # 1.0
        num_train = int(len(df_raw) * 0.7)             # 0.7
        num_test = int(len(df_raw) * 0.0)              # 0.0
        num_vali = len(df_raw) - num_train - num_test  # 0.3
        # 数据分割索引
        border1s = [0,          num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali,     len(df_raw)]
        border2s = [true_train, num_train + num_vali,     len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        # 数据切分
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # data_x 索引
        # s_begin = index
        s_begin = index * self.pred_len
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 
        #         axis = 0
        #     )
        # else:
        #     seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Test(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path,
                 pre_data=None,
                 flag='train', 
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='S', 
                 target='OT',
                 scale=True,
                 timeenc=0,
                 freq='h',
                 inverse = True,
                 cols = None):
        self.root_path = root_path
        self.data_path = data_path
        assert flag in ['test']
        type_map = {'test': 0}
        self.set_type = type_map[flag]
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        self.features = features
        self.target = target
        self.freq = freq
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        # 读取数据
        self.__read_data__()

    def __read_data__(self):
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.dropna(axis=1, how='any', inplace=True)
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 预测特征变量数据
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # 训练/测试/验证数据集分割
        # 数据分割比例
        true_train = int(len(df_raw))                  # 1.0
        num_train = int(len(df_raw) * 1.0)             # 1.0
        num_test = int(len(df_raw) * 0.0)              # 0.0
        num_vali = len(df_raw) - num_train - num_test  # 0.0
        # 数据分割索引
        border1s = [0,          num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali,     len(df_raw)]
        border2s = [true_train, num_train + num_vali,     len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        # 数据切分
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # data_x 索引
        # s_begin = index
        s_begin = index * self.pred_len
        s_end = s_begin + self.seq_len
        # logger.info(f"s_begin: {s_begin}")
        # logger.info(f"s_end: {s_end}")
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # logger.info(f"r_begin: {r_begin}")
        # logger.info(f"r_end: {r_end}")
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 
        #         axis = 0
        #     )
        # else:
        #     seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path,
                 pre_data=None, 
                 flag='pred', 
                 size=None,  # size: [seq_len, label_len, pred_len]
                 features='S',
                 target='OT', 
                 scale=True, 
                 timeenc=0, 
                 freq='15min',
                 inverse=True,
                 cols=None):
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        self.pre_data = pre_data
        # data type
        assert flag in ['pred']
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.freq = freq
        self.features = features
        self.cols = cols
        self.timeenc = timeenc
        self.target = target
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        # data read
        self.__read_data__()

    def __read_data__(self):
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.dropna(axis=1, how='any', inplace=True)
        # 因为进行预测的时候代码会自动帮我们取最后的数据我们只需做好拼接即可
        if self.pre_data is not None and not self.pre_data.empty:
            df_raw = pd.concat([df_raw, self.pre_data], axis=0, ignore_index=True)
        # 数据特征排序
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 预测特征变量数据
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        # 数据窗口索引
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        # logger.info(f"border1: {border1}")
        # logger.info(f"border2: {border2}")
        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # logger.info(f"data: \n{data}")
        # logger.info(f"data.shape: {data.shape}")
        # 时间戳特征处理
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date, format='mixed')
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), 
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        # 数据切分
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # logger.info(f"self.data_x: \n{self.data_x}")
        # logger.info(f"self.data_x.shape: {self.data_x.shape}")
        # logger.info(f"self.data_y: \n{self.data_y}")
        # logger.info(f"self.data_y.shape: {self.data_y.shape}")

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
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.detach().cpu().numpy())




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
