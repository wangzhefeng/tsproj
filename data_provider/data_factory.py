# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041902
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

from loguru import logger
from torch.utils.data import DataLoader

from data_provider.Dataset_Custom import Dataset_Custom
from data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute, 
    Dataset_M4,
    MSLSegLoader, 
    PSMSegLoader,
    SMAPSegLoader, 
    SMDSegLoader,
    SWATSegLoader, 
    UEAloader
)
from data_provider.uea import collate_fn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    'UEA': UEAloader
}


def data_provider(args, flag):
    # 数据集和数据预处理类构造
    Data = data_dict[args.data]
    # 日期时间特征编码策略
    timeenc = 0 if args.embed != 'timeF' else 1
    # 数据集参数处理
    if flag == 'test':
        freq = args.freq
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # batch_size=1 for evaluation
        shuffle_flag = False
        drop_last = True
    else:
        freq = args.freq
        batch_size = args.batch_size  # batch_size for train and valid
        shuffle_flag = True
        drop_last = True
    # 构造数据集合数据加载器
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(root_path = args.root_path, win_size = args.seq_len, flag = flag)
        logger.info(flag, len(data_set))
        data_loader = DataLoader(
            dataset = data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(root_path = args.root_path, flag = flag)
        logger.info(flag, len(data_set))
        data_loader = DataLoader(
            dataset = data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
            collate_fn = lambda x: collate_fn(x, max_len = args.seq_len)
        )
        return data_set, data_loader
    else:
        # TODO M4
        if args.data == 'm4':
            drop_last = False
        # 加载数据集
        data_set = Data(
            root_path = args.root_path,
            data_path = args.data_path,
            flag = flag,
            size = [args.seq_len, args.label_len, args.pred_len],
            features = args.features,
            target = args.target,
            timeenc = timeenc,
            freq = freq,
            seasonal_patterns = args.seasonal_patterns
        )
        logger.info(flag, len(data_set))
        # 构建数据加载器
        data_loader = DataLoader(
            dataset = data_set,
            batch_size = batch_size,
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
