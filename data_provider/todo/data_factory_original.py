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

from data_provider.data_loader_original import (
    Dataset_Custom, Dataset_ETT_hour,
    Dataset_ETT_minute, Dataset_M4,
    PSMSegLoader, MSLSegLoader,
    SMAPSegLoader, SMDSegLoader,
    SWATSegLoader, UEAloader, 
)
from data_provider.uea import collate_fn

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
    pass

if __name__ == "__main__":
    main()

