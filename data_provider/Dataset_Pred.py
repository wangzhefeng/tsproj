# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Dataset_Pred.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-21
# * Version     : 0.1.052117
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
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Dataset_Pred(Dataset):
    
    def __init__(self, 
                 root_path, 
                 flag = "pred", 
                 size = None,  # size [seq_len, label_len, pred_len]
                 features = "S", 
                 data_path = "ETTh1.csv",
                 target = "OT", 
                 scale = True, 
                 inverse = False,
                 timeenc = 0,
                 freq = "15min",
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
        assert flag in ["pred"]

        self.features = features  # TODO 'S': 单序列, "M": 多序列, "MS": 多序列
        self.target = target  # 预测目标标签
        self.scale = scale  # 是否进行标准化
        self.inverse = inverse  # ???
        self.timeenc = timeenc  # ???
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

        # data columns
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





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
