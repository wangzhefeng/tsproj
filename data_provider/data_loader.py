# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features
from utils.log_util import logger

warnings.filterwarnings('ignore')


class Dataset_Train(Dataset):
    
    def __init__(self, 
                 args,
                 root_path, 
                 data_path,
                 flag='train', 
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='MS', 
                 target='OT',
                 freq='15min',
                 timeenc=0,
                 seasonal_patterns=None,
                 scale=True,
                 inverse=True,
                 cols=None):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{30 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{30 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) 
        del df_raw["idx"]
        logger.info(f"Train data shape: {df_raw.shape}")
        # 缺失值处理
        # df_raw.dropna(axis=0, how='any', inplace=True)
        df_raw.dropna(axis=1, how='any', inplace=True)
        logger.info(f"Train data shape after drop na: {df_raw.shape}")
        # 数据特征排序
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 根据预测任务进行特征筛选
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # TODO 训练/测试/验证数据集分割: 选取当前flag下的数据
        # 数据分割比例
        all_train = int(len(df_raw))
        num_train = int(len(df_raw) * self.args.train_ratio)
        num_test = int(len(df_raw) * self.args.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
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
        # logger.info(f"Train data after standardization: \n{data} \ndata shape: {data.shape}") 
 
        # 时间特征处理
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
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
        # 数据增强
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )
        # logger.info(f"debug::data_x: \n{self.data_x} \ndata_x shape: {self.data_x.shape}")
        # logger.info(f"debug::data_y: \n{self.data_y} \ndata_y shape: {self.data_y.shape}")
        # logger.info(f"debug::data_stamp: \n{self.data_stamp} \ndata_stamp shape: {self.data_stamp.shape}")

    def __getitem__(self, index):
        # TODO data_x 索引
        if self.flag == 'test':
            s_begin = index * self.pred_len
        else:
            # TODO train
            s_begin = index
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # logger.info(f"debug::index: {index}")
        # logger.info(f"debug::s_begin:s_end {s_begin}:{s_end}")
        # logger.info(f"debug::r_begin:r_end {r_begin}:{r_end}")
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
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
                 timeenc=0, 
                 freq='15min',
                 seasonal_patterns=None,
                 scale=True, 
                 inverse=True,
                 cols=None):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        assert flag in ['pred']
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{30 * '-'}")
        logger.info(f"Load and Preprocessing data...")
        logger.info(f"{30 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        del df_raw["idx"]
        logger.info(f"Train data shape: {df_raw.shape}")
        # 缺失值处理
        df_raw.dropna(axis=1, how='any', inplace=True)
        logger.info(f"Train data shape after dropna: {df_raw.shape}") 
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
        logger.info(f"Train data after feature selection: \n{df_data.head()} \ndata shape: {df_data.shape}")  
        # 数据标准化
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        logger.info(f"Train data after standardization: \n{data} \ndata shape: {data.shape}")
        
        # 数据窗口索引
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        logger.info(f"Forecast input data index: {border1}:{border2}, data length: {border2-border1}")

        # 时间戳特征处理
        # history date
        forecast_history_stamp = df_raw[['date']][border1:border2]
        forecast_history_stamp['date'] = pd.to_datetime(forecast_history_stamp['date'], format='mixed')
        # future date
        pred_dates = pd.date_range(forecast_history_stamp['date'].values[-1], periods=self.pred_len + 1, freq=self.freq)
        self.forecast_start_time = pred_dates[1]
        # history + future date
        df_stamp = pd.DataFrame({'date': list(forecast_history_stamp['date'].values) + list(pred_dates[1:])})
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
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # self.data_y = data[border1:border2]
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
        logger.info(f"debug::index: {index}")
        logger.info(f"debug::seq_x index:      s_begin:s_end {s_begin}:{s_end}")
        logger.info(f"debug::seq_x_mark index: s_begin:s_end {s_begin}:{s_end}")
        logger.info(f"debug::seq_y index:      r_begin:(r_begin+label_len) {r_begin}:{r_begin+self.label_len}")
        logger.info(f"debug::seq_y_mark index: r_begin:r_end {r_begin}:{r_end}")
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:(r_begin+self.label_len)]
        else:
            seq_y = self.data_y[r_begin:(r_begin+self.label_len)]
        # seq_y = self.data_y[r_begin:(r_begin+self.label_len)]
        # 时间特征分割
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
