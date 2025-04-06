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

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler,
)

from utils.timefeatures import time_features
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Dataset_Train_dl:

    def __init__(self, 
                 args,
                #  root_path=None,
                #  data_path=None,
                 flag="train",
                #  size=None,
                #  features="MS",
                #  target="WIND",
                #  timeenc=0,
                #  freq="d",
                #  seasonal_patterns=None,
                #  scale=True,
                #  inverse=True,
                #  cols=None
                 ) -> None:
        self.args = args
        # task flag
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        """
        self.root_path = root_path
        self.data_path = data_path 
        
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        
        self.features = features
        self.target = target
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns
        self.timeenc = timeenc
        
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        
        # 读取数据
        self._read_data()
        """

    def _read_data(self):
        """
        data read
        """
        # TODO
        # logger.info(f'{30 * '-'}')
        # logger.info(f"Load and Preprocessing {self.args.flag} data...")
        # logger.info(f'{30 * '-'}')
        
        # 读取数据文件
        df_raw = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        logger.info(f"Train data shape: {df_raw.shape}")
        
        # 缺失值处理
        df_raw.dropna(axis=0, how='any', inplace=True)
        logger.info(f"Train data shape after drop na: {df_raw.shape}")
        
        # 数据特征重命名
        df_raw.columns = [col.lower() for col in df_raw.columns]
        self.args.target = self.args.target.lower()
        
        # 数据特征排序
        cols_names = list(df_raw.columns)
        cols_names.remove(self.args.target)
        cols_names.remove('date')
        df_raw = df_raw[['date'] + cols_names + [self.args.target]]
        # 根据预测任务进行特征筛选
        if self.args.features == 'M' or self.args.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.args.features == 'S':
            df_data = df_raw[[self.args.target]]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")

        """
        # 训练/测试/验证数据集分割: 选取当前flag下的数据
        # 数据分割比例
        num_train = int(len(df_raw) * self.args.train_ratio)  # 0.7
        num_test = int(len(df_raw) * self.args.test_ratio)    # 0.2
        num_vali = len(df_raw) - num_train - num_test  # 0.1
        logger.info(f"Train data length: {num_train}, Valid data length: {num_vali}, Test data length: {num_test}")
        # 数据分割索引
        border1s = [0,         num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali,     len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}, data length: {border2-border1}")

        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values 
        logger.info(f"Train data after standardization: \n{data} \ndata shape: {data.shape}")
 
        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        # logger.info(f"Forecast input data_stamp: \n{data_stamp} \ndata_stamp shape: {data_stamp.shape}")
        
        # 数据切分
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # TODO 数据增强
        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
        #         self.data_x, self.data_y, self.args
        #     )
        """

        return df_data

    def _transform_data(self, df):
        """
        data scaler
        """
        if self.args.scale:
            self.scaler = StandardScaler()
            self.scaler.fit_transform(df.values)
            data = self.scaler.transform(df.values)
        else:
            data = np.array(df)
        
        return data
    
    # def __getitem__(self, index):
    #     pass
   
    # def __len__(self):
    #     return len(self.data_x) - self.seq_len - self.pred_len + 1

    def _inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def RecursiveMultiStep(self, data):
        """
        递归多步预测(单步滚动预测)
        
        例如：多变量：123456789 => 12345-67、23456-78、34567-89...
        例如：单变量：123456789 => 123-4、234-5、345-6...
        """
        data = np.array(data)
        data_X = []  # 保存 X
        data_Y = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - self.args.seq_len):  # [0, len(data)-seq_len - 1]
            data_X.append(data[index:(index + self.args.seq_len)])
            data_Y.append(data[index + self.args.seq_len][-1])
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)
        logger.info(f"data_X: \n{data_X} \ndata_X shape: {data_X.shape}")
        logger.info(f"data_Y: \n{data_Y} \ndata_Y shape: {data_Y.shape}")
        # 训练/测试/验证数据集分割: 选取当前flag下的数据
        # 数据分割比例
        data_len = data_X.shape[0]
        num_train = int(data_len * self.args.train_ratio)  # 0.7
        num_test = int(data_len * self.args.test_ratio)    # 0.2
        num_vali = data_len - num_train - num_test         # 0.1
        logger.info(f"Train data length: {num_train}, Valid data length: {num_vali}, Test data length: {num_test}")
        # 划分训练集、测试集
        border1s = [0,         num_train,            num_train + num_vali]
        border2s = [num_train, num_train + num_vali, data_len]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        data_x = data_X[border1:border2, :].reshape(-1, self.args.seq_len, self.args.feature_size)  # (batch_size, seq_len, feature_size)
        data_y = data_Y[border1:border2].reshape(-1, 1)  # (batch_size, num_target) 
        logger.info(f"data_x: \n{data_x} \ndata_x shape: {data_x.shape}")
        logger.info(f"data_y: \n{data_y} \ndata_y shape: {data_y.shape}")
        
        return data_x, data_y
    
    def DirectMultiStepOutput(self, data):
        """
        直接多步预测
        
        例如：123456789 => 12345-67、23456-78、34567-89...
        """
        dataX = []  # 保存 X
        dataY = []  # 保存 Y
        # 将整个窗口的数据保存到 X 中，将未来一个时刻的数据保存到 Y 中
        for index in range(len(data) - self.args.seq_len - 1):
            dataX.append(data[index:(index + self.args.seq_len)])
            dataY.append(data[(index + self.args.seq_len):(index + self.args.seq_len + self.args.output_size)][:, -1].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集长度
        train_size = int(np.round(self.args.train_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.args.seq_len, self.args.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.args.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.args.seq_len, self.args.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.args.output_size)
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
        for index in range(len(data) - self.args.seq_len - 1):
            dataX.append(data[index:(index + self.args.seq_len)][:, -1])
            dataY.append(data[(index + self.args.seq_len):(index + self.args.seq_len + self.args.output_size)][:, -1].tolist())
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        # 训练集大
        train_size = int(np.round(self.args.train_ratio * dataX.shape[0]))
        # 划分训练集、测试集
        self.x_train = dataX[:train_size, :].reshape(-1, self.args.seq_len, self.args.feature_size)
        self.y_train = dataY[:train_size].reshape(-1, self.args.output_size)
        self.x_test = dataX[train_size:, :].reshape(-1, self.args.seq_len, self.args.feature_size)
        self.y_test = dataY[train_size:].reshape(-1, self.args.output_size)
        # 创建 torch Dataset 和 DataLoader 
        [train_data, train_loader, test_data, test_loader] = self._dataset_dataloader()
        
        return [train_data, train_loader, test_data, test_loader]

    def run(self):
        # 读取数据
        data = self._read_data()
        # data = data.head(5)
        logger.info(f"data: \n{data} \ndata shape: {data.shape}")
        
        # 数据预处理
        data = self._transform_data(data)
        logger.info(f"data after transform: \n{data} \ndata shape after transform: {data.shape}")
        
        # 选择预测方法
        if self.args.pred_method == "recursive_multi_step":  # 递归多步预测
            data_x, data_y = self.RecursiveMultiStep(data)
        elif self.args.pred_method == "direct_multi_step_output":  # 直接多步预测
            data_x, data_y = self.DirectMultiStepOutput(data)
        elif self.args.pred_method == "direct_recursive_mix":  # 直接递归多步混合
            data_x, data_y = self.DirectRecursiveMix(data)

        return data_x, data_y


class Dataset_Pred_dl():
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
