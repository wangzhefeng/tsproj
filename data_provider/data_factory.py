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
from data_provider.data_loader import (
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
    """
    构造 torch 数据集和数据加载器

    Args:
        args (_type_): 命令行参数
        flag (_type_):  任务类型, flat: ["train", "test", "val"]

    Returns:
        _type_: torch Dataset, DataLoader
    """
    # ------------------------------
    # 数据集和数据预处理类构造
    # ------------------------------
    Data = data_dict[args.data]
    # ------------------------------
    # 数据集和数据加载器参数
    # ------------------------------
    if flag == 'test':
        freq = args.freq  # 序列频率
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size  # batch_size for evaluation in ad and clf
        else:
            batch_size = 1  # batch_size=1 for evaluation
        shuffle_flag = False  # 是否进行 shuffle
        drop_last = True  # 是否丢掉最后一个点
    else:
        freq = args.freq  # 序列频率
        batch_size = args.batch_size  # batch_size for train and valid
        shuffle_flag = True  # 是否进行 shuffle
        drop_last = True  # 是否丢掉最后一个点
    
    # ------------------------------
    # 构造数据集合数据加载器
    # ------------------------------
    if args.task_name == 'anomaly_detection':
        # data set
        data_set = Data(root_path = args.root_path, win_size = args.seq_len, flag = flag)
        logger.info(f"{LOGGING_LABEL}.data_provider, {flag}: {len(data_set)}")
        # data loader
        drop_last = False
        data_loader = DataLoader(
            dataset = data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        # data set
        data_set = Data(root_path = args.root_path, flag = flag)
        logger.info(f"{LOGGING_LABEL}.data_provider, {flag}: {len(data_set)}")
        # data loader
        drop_last = False
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
        # 加载数据集
        timeenc = 0 if args.embed != 'timeF' else 1  # 日期时间特征编码策略
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
        logger.info(f"{LOGGING_LABEL}.data_provider, {flag}: {len(data_set)}")
        # 构建数据加载器
        if args.data == 'm4':  # M4 特殊处理
            drop_last = False
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
