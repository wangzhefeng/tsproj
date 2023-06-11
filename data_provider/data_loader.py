# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-04
# * Version     : 0.1.050418
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
import glob
import pickle
import re
import warnings
from typing import List

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_provider.data_splitor import split_data
from utils.timefeatures import time_features
from utils.timestamp_utils import to_unix_time
# from utils.tools import StandardScaler

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Dataset_Custom(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path = "ETTh1.csv",
                 flag = "train", 
                 size = None,  # [seq_len, label_len, pred_len]
                 freq = "h",
                 features = "S",
                 cols = None,
                 timeenc = 0,
                 target = "OT",
                 train_ratio = 0.7,
                 test_ratio = 0.2,
                 scale = True, 
                 inverse = False, 
                 seasonal_patterns = "Yearly") -> None:
        # data file path
        self.root_path = root_path  # 数据根路径
        self.data_path = data_path  # 数据文件路径
        # data type
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        # data size
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # data freq, feature columns, and target
        self.freq = freq  # 频率
        self.features = features  # 特征类型 'S': 单序列, "M": 多序列, "MS": 多序列
        self.cols = cols  # 表列名
        self.timeenc = timeenc  # 时间特征
        self.target = target  # 预测目标标签
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        # data preprocess 
        self.scale = scale # 是否进行标准化
        self.inverse = inverse  # 是否逆转换
        self.seasonal_patterns = seasonal_patterns  # 季节模式
        # data read
        self.__read_data__()
    
    def __read_data__(self):
        """
        Returns:
            data_stamp: 日期时间动态特征
            data_x: # TODO
            data_y: # TODO
        """
        # ------------------------------
        # data read
        # df_raw: (date, features, target)
        # df_data: (features, target) or (target)
        # ------------------------------
        # data read(df_raw.columns: ["date", ...(other features), target feature])
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # data column sort
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw["date"] + cols + [self.target]
        # 根据数据格式进行数据处理
        if self.features == "M" or self.features == "MS":  # 多序列(date, feature1, feature2, feature3, ..., target)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]  # 不包含 'date' 列的预测特征列(包含 target)
        elif self.features == "S":  # 单序列(date, target)
            df_data = df_raw[[self.target]]  # 不包含 'date' 列的预测标签列(target)
        # ------------------------------
        # train/val/test split
        # ------------------------------
        # train/val/test 长度
        num_df_raw = len(df_raw)
        num_train = int(num_df_raw * self.train_ratio)
        num_test = int(num_df_raw * self.test_ratio)
        num_val = num_df_raw - num_train - num_test
        #   - train: 0 : num_train
        #   - val: (num_train - seq_len) : (num_train + num_val)
        #   - test: (len_df_raw - num_test - seq_len) : len_df_raw
        border1s = [0, num_train - self.seq_len, num_df_raw - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, num_df_raw]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # ------------------------------
        # 预测特征(features)和预测标签(target)标准化
        # data: (features, target) or (target)
        # ------------------------------
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)  # 标准化
        else:
            data = df_data.values  # 非标准化
        # ------------------------------
        # 特征构造(train/test/val)
        # [self.data_stamp:, self.data_x] -> [time_features, features, target]
        # ------------------------------
        # 日期时间动态特征
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc = self.timeenc, freq = self.freq)
        self.data_stamp = data_stamp
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        self.data_x = data[border1:border2]
        # ------------------------------
        # 根据是否进行标准化逆转换返回数据
        # ------------------------------
        if self.inverse:
            self.data_y = df_data.values[border1:border2]  # 非标准化
        else:
            self.data_y = data[border1:border2]  # 标准化
    
    def __getitem__(self, index):
        # history
        s_begin = index
        s_end = s_begin + self.seq_len
        # target history and predict
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        seq_x = self.data_x[s_begin:s_end]
        # ???
        if self.inverse:
            seq_y = np.concatenate([
                self.data_x[r_begin:(r_begin + self.label_len)],
                self.data_y[(r_begin + self.label_len):r_end],
            ], axis = 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # 日期时间动态特征
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
                 data_path = "ETTh1.csv",
                 flag = "pred", 
                 size = None,
                 features = "S", 
                 target = "OT", 
                 scale = True, 
                 inverse = False,
                 timeenc = 0,
                 freq = "15min",
                 cols = None) -> None:
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # data type
        assert flag in ["pred"]
        # data feature and target
        self.features = features  # 'S': 单序列, "M": 多序列, "MS": 多序列
        self.target = target  # 预测目标标签
        # data preprocess
        self.scale = scale  # 是否进行标准化
        self.inverse = inverse
        self.timeenc = timeenc
        # 其他
        self.freq = freq  # 频率
        self.cols = cols  # 表列名
        self.root_path = root_path  # 根路径
        self.data_path = data_path  # 数据路径
        self.__read_data__()  # 数据读取
    
    def __read_data__(self):
        """
        Returns:
            data_stamp: 日期时间动态特征
            data_x: # TODO
            data_y: # TODO
        """
        # data read(df_raw.columns: ["date", ...(other features), target feature])
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # data columns sort
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw["date"] + cols + [self.target]
        # 根据数据格式进行数据处理
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]  # 不包含 'date' 列的预测特征列
        elif self.features == "S":
            df_data = df_raw[[self.target]]  # 不包含 'date' 列的预测标签列

        # train/val/test 索引
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # 特征和预测标签标准化(不包含 'date' 列)
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)  # 标准化
        else:
            data = df_data.values  # 非标准化
        # ------------------------------
        # 特征构造
        # ------------------------------
        # 日期时间动态特征
        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods = self.pred_len + 1, freq = self.freq)
        df_stamp = pd.DataFrame(columns = ["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc = self.timeenc, freq = self.freq[-1:])
        self.data_stamp = data_stamp
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        self.data_x = data[border1:border2]
        # ?
        if self.inverse:
            self.data_y = df_data.values[border1:border2]  # 不包含 'date' 列的预测特征列或预测标签
        else:
            self.data_y = data[border1:border2]  # ???
     
    def __getitem__(self, index):
        # history
        s_begin = index
        s_end = s_begin + self.seq_len
        # target history and predict
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        seq_x = self.data_x[s_begin:s_end]
        # ???
        if self.inverse:
            seq_y = self.data_x[r_begin:(r_begin + self.label_len)]
        else:
            seq_y = self.data_y[r_begin:(r_begin + self.label_len)]
        # 日期时间动态特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
            
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# long-term forecasting / imputation
class Dataset_ETT_hour(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path = "ETTh1.csv",
                 flag = "train", 
                 size = None,
                 features = "S", 
                 target = "OT", 
                 scale = True, 
                 inverse = False,
                 timeenc = 0,
                 freq = "h",
                 cols = None) -> None:
        # size info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]  # data type
        self.features = features  # 'S': 单序列, "M": 多序列, "MS": 多序列
        self.target = target  # 预测目标标签
        self.scale = scale  # 是否进行标准化
        self.inverse = inverse 
        self.timeenc = timeenc
        self.freq = freq  # 频率
        self.cols = cols  # 表列名
        self.root_path = root_path  # 根路径
        self.data_path = data_path  # 数据路径
        self.__read_data__()  # 数据读取
    
    def __read_data__(self):
        # data read(df_raw.columns: ["date", ...(other features), target feature])
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 根据数据格式进行数据处理
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]  # 不包含 'date' 列的预测特征列
        elif self.features == "S":
            df_data = df_raw[[self.target]]  # 不包含 'date' 列的预测标签列
        
        # ??? train/val/test 索引
        num_train = 12*30*24
        num_test = None
        num_val = 4*30*24
        border1s = [0, num_train - self.seq_len, 12*30*24 + 4*30*24]
        border2s = [num_train, num_train + num_val, 12*30*24 + 8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 特征和预测标签标准化(不包含 'date' 列)
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)  # 标准化
        else:
            data = df_data.values  # 非标准化
        # ------------------------------
        # 特征构造
        # ------------------------------
        # 日期时间动态特征
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc = self.timeenc, freq = self.freq)
        self.data_stamp = data_stamp
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        self.data_x = data[border1:border2]
        # ???
        if self.inverse:
            self.data_y = df_data.values[border1:border2]  # 不包含 'date' 列的预测特征列或预测标签
        else:
            self.data_y = data[border1:border2]  # ???
    
    def __getitem__(self, index):
        # history
        s_begin = index
        s_end = s_begin + self.seq_len
        # target history and predict
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        seq_x = self.data_x[s_begin:s_end]
        # ???
        if self.inverse:
            seq_y = np.concatenate([
                self.data_x[r_begin:(r_begin + self.label_len)],
                self.data_y[(r_begin + self.label_len):r_end],
            ], axis = 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # 日期时间动态特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
            
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# long-term forecasting / imputation
class Dataset_ETT_minute(Dataset):
    
    def __init__(self, 
                 root_path, 
                 data_path = "ETTm1.csv",
                 flag = "train", 
                 size = None,
                 features = "S", 
                 target = "OT", 
                 scale = True, 
                 inverse = False,
                 timeenc = 0,
                 freq = "t",
                 cols = None) -> None:
        # size info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]  # data type
        self.features = features  # 'S': 单序列, "M": 多序列, "MS": 多序列
        self.target = target  # 预测目标标签
        self.scale = scale  # 是否进行标准化
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq  # 频率
        self.cols = cols  # 表列名
        self.root_path = root_path  # 根路径
        self.data_path = data_path  # 数据路径
        self.__read_data__()  # 数据读取
    
    def __read_data__(self):
        """
        Returns:
            data_stamp: 日期时间动态特征
            data_x: # TODO
            data_y: # TODO
        """
        # data read(df_raw.columns: ["date", ...(other features), target feature])
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 根据数据格式进行数据处理
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]  # 不包含 'date' 列的预测特征列
        elif self.features == "S":
            df_data = df_raw[[self.target]]  # 不包含 'date' 列的预测标签列
        
        # train/val/test 长度
        num_train = 12*30*24*4
        num_test = 4*30*24*4
        num_val = 4*30*24*4
        # train/val/test 索引
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 特征和预测标签标准化(不包含 'date' 列)
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)  # 标准化
        else:
            data = df_data.values  # 非标准化
        # ------------------------------
        # 特征构造
        # ------------------------------
        # 日期时间动态特征
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc = self.timeenc, freq = self.freq)
        self.data_stamp = data_stamp
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        self.data_x = data[border1:border2]
        # ???
        if self.inverse:
            self.data_y = df_data.values[border1:border2]  # 不包含 'date' 列的预测特征列或预测标签
        else:
            self.data_y = data[border1:border2]
        
    def __getitem__(self, index):
        # history
        s_begin = index
        s_end = s_begin + self.seq_len
        # target history and predict
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 预测特征和预测标签(不包含日期时间特征、'date' 列)
        seq_x = self.data_x[s_begin:s_end]
        # ???
        if self.inverse:
            seq_y = np.concatenate([
                self.data_x[r_begin:(r_begin + self.label_len)],
                self.data_y[(r_begin + self.label_len):r_end],
            ], axis = 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # 日期时间动态特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
            
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# short-term forecasting
from m4 import M4Dataset, M4Meta
class Dataset_M4(Dataset):
    
    def __init__(self, 
                 root_path, 
                 # data_path = 'ETTh1.csv',
                 flag = 'pred', 
                 size = None,
                 features = 'S', 
                 target = 'OT', 
                 scale = False, 
                 inverse = False, 
                 timeenc = 0, 
                 # freq = '15min',
                 seasonal_patterns = 'Yearly'):
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # TODO
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag
        # 数据读取
        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training = True, dataset_file = self.root_path)
        else:
            dataset = M4Dataset.load(training = False, dataset_file = self.root_path)
        training_values = np.array([v[~np.isnan(v)] for v in dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low = max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high = len(sampled_timeseries),
            size = 1
        )[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


# anomaly detection
class PSMSegLoader(Dataset):

    def __init__(self, root_path, win_size, step = 1, flag = "train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# anomaly detection
class MSLSegLoader(Dataset):

    def __init__(self, root_path, win_size, step = 1, flag = "train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# anomaly detection
class SMAPSegLoader(Dataset):

    def __init__(self, root_path, win_size, step = 1, flag = "train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# anomaly detection
class SMDSegLoader(Dataset):

    def __init__(self, root_path, win_size, step = 100, flag = "train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# anomaly detection
class SWATSegLoader(Dataset):

    def __init__(self, root_path, win_size, step = 1, flag = "train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# classification
from sktime.datasets import load_from_tsfile_to_dataframe
from uea import Normalizer, interpolate_missing, subsample
class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, 
                 root_path, 
                 file_list = None, 
                 limit_size = None, 
                 flag = None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list = file_list, flag = flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list = None, flag = None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]

        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]

        if len(input_paths) == 0:
            pattern = None  # TODO
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))
        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath, 
            return_separate_X_and_y = True,
            replace_missing_vals_with = 'NaN'
        )
        labels = pd.Series(labels, dtype = "category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype = np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]
        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns}) \
                    .reset_index(drop = True) \
                    .set_index(pd.Series(lengths[row, 0] * [row])) 
                for row in range(df.shape[0])
            ), 
            axis = 0
        )
        # Replace NaN values
        grp = df.groupby(by = df.index)
        df = grp.transform(interpolate_missing)
        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim = True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim = 1, keepdim = True, unbiased = False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


# !paddlepaddle
import paddle
class TSDataset(paddle.io.Dataset):
    """
    时序 Dataset

    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    """
    
    def __init__(self,
                 data,
                 ts_col: str,
                 use_cols: List,
                 labels: List,
                 input_len: int,  # 24*4*5
                 pred_len: int,  # 24*4
                 stride: int,
                 data_type: str = "train",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15) -> None:
        super(TSDataset, self).__init__()
        # data features
        self.ts_col = ts_col  # 时间戳列
        self.use_cols = use_cols  # 训练时使用的特征列
        self.labels = labels  # 待预测的标签列
        # data len
        self.input_len = input_len  # 模型输入数据的样本点长度，15分钟间隔，一个小时4个点，近5天的数据就是:24*4*5个点
        self.pred_len = pred_len  # 预测长度，预测次日00:00至23:45实际功率，即1天:24*4个点
        self.stride = stride  # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率，所以x和label要间隔:19*4个点
        # data type
        assert data_type in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.data_type = data_type  # 需要加载的数据类型
        self.set_type = type_map[self.data_type]
        # data split
        self.train_ratio= train_ratio  # 训练集划分比例
        self.val_ratio = val_ratio  # 验证集划分比例
        # data transform
        self.scale = True  # 是否需要标准化

        self.transform(data)
     
    def transform(self, df):
        # 获取unix 时间戳、输入特征、预测标签
        time_stamps = df[self.ts_col].apply(lambda x: to_unix_time(x)).values
        x_values = df[self.use_cols].values
        y_values = df[self.labels].values
        # 划分数据集
        num_train = int(len(df) * self.train_ratio)
        num_val = int(len(df) * self.val_ratio)
        num_test = len(df) - num_train - num_val
        border1s = [0, num_train - self.input_len - self.stride, len(df) - num_test - self.input_len - self.stride]
        border2s = [num_train, num_train + num_val, len(df)]
        # 获取 data_type 下的左右数据截取边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 使用训练集得到 scaler 对象
            train_data = x_values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(x_values)
            # 保存 scaler
            pickle.dump(self.scaler, open("scaler.pkl", "wb"))
        else:
            data = x_values
        # array to paddle tensor
        self.time_stamps = paddle.to_tensor(time_stamps[border1:border2], dtype = "int64")
        self.data_x = paddle.to_tensor(data[border1:border2], dtype = "float32")
        self.data_y = paddle.to_tensor(y_values[border1:border2], dtype = "float32") 

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


# !paddlepaddle
import paddle
class TSPredDataset(paddle.io.Dataset):
    """
    时序预测 Dataset 

    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    """

    def __init__(self) -> None:
        super(TSPredDataset, self).__init__()


class Data_Loader:
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


class Data_LoaderV2:
 
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


# TODO
from pandas.tseries import to_offset
def data_preprocess(df):
    """
    数据预处理
    1. 数据排序
    2. 去除重复值
    3. 重采样（ 可选）
    4. 缺失值处理
    5. 异常值处理
    """
    # 排序
    df = df.sort_values(by = "DATATIME", ascending = True)
    logger.info(f"df.shape: {df.shape}")
    logger.info(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")
    # 去除重复值
    df = df.drop_duplicates(subset = "DATATIME", keep = "first")
    logger.info(f"After dropping dulicates: {df.shape}")
    # 重采样（可选）+ 缺失值处(理线性插值)：比如 04 风机缺少 2022-04-10 和 2022-07-25 两天的数据，重采样会把这两天数据补充进来
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.set_index("DATATIME")
    df = df.resample(rule = to_offset('15T').freqstr, label = 'right', closed = 'right')
    df = df.interpolate(method = 'linear', limit_direction = 'both').reset_index()
    # 异常值处理
    # 当实际风速为 0 时，功率设置为 0
    df.loc[df["ROUND(A.WS,1)"] == 0, "YD15"] = 0
    # TODO 风速过大但功率为 0 的异常：先设计函数拟合出：实际功率=f(风速)，然后代入异常功率的风速获取理想功率，替换原异常功率
    # TODO 对于在特定风速下的离群功率（同时刻用 IQR 检测出来），做功率修正（如均值修正）

    return df


# data and preprocess class 
data_dict = {
    'custom': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'm4': Dataset_M4,
    'MSL': MSLSegLoader,
    'PSM': PSMSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
}




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
