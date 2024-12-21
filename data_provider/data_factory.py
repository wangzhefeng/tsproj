# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-03
# * Version     : 0.1.110300
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

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom, Dataset_ETT_hour,
    Dataset_ETT_minute, Dataset_M4,
    PSMSegLoader, MSLSegLoader,
    SMAPSegLoader, SMDSegLoader,
    SWATSegLoader, UEAloader
)
from data_provider.uea import collate_fn
from data_provider.data_loader import Data_Loader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


data_dict = {
    'custom': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute, 
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # ------------------------------
    # Basic Neural Network dataset
    # ------------------------------
    "Data_Loader": Data_Loader,
}


def data_provider(args, flag: str):
    """
    数据集准备

    Args:
        args (Dcit): 参数集
        flag (str): 任务标签, "train", "val", "test"

    Returns:
        _type_: data_set, data_loader
    """
    # 数据集类
    Data = data_dict[args.data] 
    # 区别在 test 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    # 是否丢弃最后一个 batch
    drop_last = False
    
    # 构建 Dataset 和 DataLoader
    if args.task_name == 'anomaly_detection':
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            win_size = args.seq_len,
            flag = flag,
        )
        print(f"{flag}, data_set len: {len(data_set)}")
        # DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size = args.batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            flag = flag,
        )
        print(f"{flag}, data_set len: {len(data_set)}")
        # DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size = args.batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
            collate_fn = lambda x: collate_fn(x, max_len = args.seq_len)
        )
        return data_set, data_loader
    else:
        # 是否对时间戳进行编码
        timeenc = 0 if args.embed != 'timeF' else 1
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            data_path = args.data_path,
            flag = flag,
            size = [args.seq_len, args.label_len, args.pred_len],
            features = args.features,
            target = args.target,
            freq = args.freq,
            seasonal_patterns = args.seasonal_patterns,
            scale = args.scale,
            timeenc = timeenc,
        )
        print(f"{flag}, data_set len: {len(data_set)}")
        # DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size = args.batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
        )
        return data_set, data_loader




# 测试代码 main 函数
def main():
    import pandas as pd
    import torch

    df = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", end="2024-03-20", freq="d"),
        "OT": range(1, 81),
    })
    print(f"raw df: {df}")
    
    class Config:
        embed = "time_F"
        root_path = None
        data_path = None
        seq_len = 6
        label_len = 2
        pred_len = 2
        features = "S"
        target = "OT"
        freq = "d"
        seasonal_patterns = None
        scale = False
        augmentation_ratio = 0
        batch_size = 1
        num_workers = 1
    args = Config()
    
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != 'timeF' else 1
    # 任务类型
    flag = "train"
    # Dataset
    data_set = Dataset_Custom(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        flag = flag,
        size = [args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        freq = args.freq,
        seasonal_patterns = args.seasonal_patterns,
        scale = args.scale,
        timeenc = timeenc,
        df_test=df,
    )
    print(f"window sample: {len(data_set)}")

    # DataLoader
    # 区别在 test 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    # 是否丢弃最后一个 batch
    drop_last = False
    data_loader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        shuffle = shuffle_flag,
        # num_workers = args.num_workers,
        drop_last = drop_last,
    )

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        print(f"batch_x:\n {batch_x}")
        print(f"batch_y:\n {batch_y}")
        # print(batch_x_mark)
        # print(batch_y_mark)
        # decoder input
        print(batch_y[:, -args.pred_len:, :])
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        print(dec_inp)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim = 1).float()
        print(dec_inp)
        break

if __name__ == "__main__":
    main()
