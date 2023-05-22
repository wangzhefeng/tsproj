# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TSDataset.py
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
import pickle
from typing import List

from sklearn.preprocessing import StandardScaler
import paddle

from utils.tools import StandardScaler
from utils.timestamp_utils import to_unix_time

import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# !paddlepaddle
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





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
