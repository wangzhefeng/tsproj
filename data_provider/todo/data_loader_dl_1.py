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

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.ts.timefeatures import time_features
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Dataset_Train:

    def __init__(self, 
                 args,
                 root_path,
                 data_path,
                 flag="train",
                 # size=None,  # size [seq_len, label_len, pred_len]
                 seq_len=1,
                 feature_size=1,
                 output_size=1,
                 target="WIND",
                 features="MS",
                 train_ratio=0.8,
                 test_ratio=0.2,
                 pred_method="recursive_multi_step",
                 # freq="d",
                 # timeenc=0,
                 # seasonal_patterns=None,
                 scale=True,
                 inverse=True):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data flag
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        # data size
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.output_size = output_size
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.pred_method = pred_method
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        # data read
        # 读取数据
        self.data_x, self.data_y = self.__read_data__()
        # data = data.head(21)

    def __read_data__(self):
        """
        data read
        """
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 读取数据文件
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.info(f"Train data: \n{df_raw.head(10)} \nTrain data shape: {df_raw.shape}")
        # 缺失值处理
        df_raw.dropna(axis=0, how='any', inplace=True)
        logger.info(f"Train data shape after drop na: {df_raw.shape}")
        # 数据特征重命名
        df_raw.columns = [col.lower() for col in df_raw.columns]
        self.target = self.target.lower()
        # 数据特征排序
        cols_names = list(df_raw.columns)
        cols_names.remove(self.target)
        cols_names.remove('date')
        df_raw = df_raw[['date'] + cols_names + [self.target]]
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        logger.info(f"Train data columns after feature order: \n{df_raw.columns}")
        # 根据预测任务进行特征筛选
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # 数据标准化
        df_data = self._transform_data(df_data)
        logger.info(f"Train data shape after standardization: {df_data.shape}")
        # 选择预测方法
        if self.pred_method == "recursive_multi_step":  # 递归多步预测
            data_X, data_Y = self.RecursiveMultiStep(df_data)
        elif self.pred_method == "direct_multi_output":  # 直接多步预测
            data_X, data_Y = self.DirectMultiOutput(df_data)
        elif self.pred_method == "direct_recursive_multi_step_mix":  # 直接递归多步混合
            data_X, data_Y = self.DirectRecursiveMultiStepMix(df_data)
        '''
        # 时间特征处理
        df_stamp = df_data[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp
        logger.info(f"Forecast input timestamp features shape: {data_stamp.shape}")
        logger.info(f"debug::data_stamp: \n{self.data_stamp} \ndata_stamp shape: {self.data_stamp.shape}")
        '''
        # 数据分割比例
        data_len = data_X.shape[0]
        num_train = int(np.round(data_len * self.train_ratio))  # 0.7
        num_test = int(np.round(data_len * self.test_ratio))  # 0.2
        num_vali = data_len - num_train - num_test  # 0.1
        logger.info(f"Train data length: {num_train}, Valid data length: {num_vali}, Test data length: {num_test}")
        # 数据集分割
        border1s = [0,         num_train,            num_train + num_vali]
        border2s = [num_train, num_train + num_vali, data_len]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        data_x = data_X[border1:border2, :].reshape(-1, self.seq_len, self.feature_size)  # (batch_size, seq_len, feature_size)
        data_y = data_Y[border1:border2].reshape(-1, self.output_size)  # (batch_size, num_target)
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}, data length: {border2-border1}")
        logger.info(f"data_x: \n{data_x} \ndata_x shape: {data_x.shape}")
        logger.info(f"data_y: \n{data_y} \ndata_y shape: {data_y.shape}")
        
        return data_x, data_y
        
    def _transform_data(self, df):
        """
        data scaler
        """
        self.scaler = StandardScaler()
        self.scaler_target = StandardScaler()
        if self.scale: 
            # train_data = df_data[border1s[0]:border2s[0]]
            # self.scaler.fit(train_data.values)
            data = self.scaler.fit_transform(df.values)
            self.scaler_target.fit_transform(np.array(df[self.target]).reshape(-1, 1))
        else:
            data = df.values
        
        return data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def inverse_target(self, target):
        return self.scaler_target.inverse_transform(target.detach().numpy().reshape(-1, 1))
    
    def RecursiveMultiStep(self, data):
        """
        递归多步预测(单步滚动预测)
        单步预测
        """
        # 将 data 转换为 np.array
        data = np.array(data)
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        data_X, data_Y = [], []
        for index in range(len(data) - self.seq_len):  # [0, len(data)-seq_len-1]
            data_X.append(data[index:(index + self.seq_len)])
            data_Y.append(data[index + self.seq_len][-1])
        data_X, data_Y = np.array(data_X), np.array(data_Y)
        logger.info(f"data_X: \n{data_X} \ndata_X shape: {data_X.shape}")
        logger.info(f"data_Y: \n{data_Y} \ndata_Y shape: {data_Y.shape}")
        
        return data_X, data_Y
    
    def DirectMultiOutput(self, data):
        """
        直接多输出
        """
        # 将 data 转换为 np.array
        data = np.array(data)
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        data_X, data_Y = [], []
        for index in range(len(data) - self.seq_len - 1):
            data_x = data[index:(index + self.seq_len)]
            data_X.append(data_x)
            data_y = data[(index + self.seq_len):(index + self.seq_len + self.output_size)][:, -1].tolist()
            if len(data_y) == self.output_size:
                data_Y.append(data_y)
            else:
                data_X = data_X[:-1]
        data_X, data_Y = np.array(data_X), np.array(data_Y)
        logger.info(f"data_X: \n{data_X} \ndata_X shape: {data_X.shape}")
        logger.info(f"data_Y: \n{data_Y} \ndata_Y shape: {data_Y.shape}")
        
        return data_X, data_Y
    
    def DirectMultiStep(self, data):
        """
        直接多输出
        """
        # 将 data 转换为 np.array
        data = np.array(data)
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        data_X, data_Y = [], []
        for index in range(len(data) - self.seq_len - 1):
            data_x = data[index:(index + self.seq_len)]
            data_X.append(data_x)
            data_y = data[(index + self.seq_len):(index + self.seq_len + self.output_size)][:, -1].tolist()
            if len(data_y) == self.output_size:
                data_Y.append(data_y)
            else:
                data_X = data_X[:-1]
        data_X, data_Y = np.array(data_X), np.array(data_Y)
        logger.info(f"data_X: \n{data_X} \ndata_X shape: {data_X.shape}")
        logger.info(f"data_Y: \n{data_Y} \ndata_Y shape: {data_Y.shape}")
        
        return data_X, data_Y

    def DirectRecursiveMultiStepMix(self, data):
        """
        直接递归混合预测(多模型滚动预测)
        
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        # 将 data 转换为 np.array
        data = np.array(data)
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        data_X, data_Y = [], []
        for index in range(len(data) - self.seq_len - 1):
            data_x = data[index:(index + self.seq_len)]#[:, -1]  # TODO
            data_X.append(data_x)
            data_y = data[(index + self.seq_len):(index + self.seq_len + self.output_size)][:, -1].tolist()
            if len(data_y) == self.output_size:
                data_Y.append(data_y)
            else:
                data_X = data_X[:-1]
        data_X, data_Y = np.array(data_X), np.array(data_Y)
        logger.info(f"data_X: \n{data_X} \ndata_X shape: {data_X.shape}")
        logger.info(f"data_Y: \n{data_Y} \ndata_Y shape: {data_Y.shape}")
        
        return data_X, data_Y

    """
    def __getitem__(self, index):
        # data_x 索引
        if self.flag in ["train", "val"]:
            s_begin = index
        elif self.flag == "test":
            s_begin = index * self.pred_len
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
        # log
        # logger.info(f"debug::index: {index}")
        # logger.info(f"debug::seq_x index:      s_begin:s_end {s_begin}:{s_end}")
        # logger.info(f"debug::seq_x_mark index: s_begin:s_end {s_begin}:{s_end}")
        # logger.info(f"debug::seq_y index:      r_begin:r_end {r_begin}:{r_end}")
        # logger.info(f"debug::seq_y_mark index: r_begin:r_end {r_begin}:{r_end}")
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    """


class Dataset_Pred():
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
