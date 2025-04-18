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

import torch
from torch.utils.data import TensorDataset, DataLoader

from data_provider.data_loader_dl import (
    Dataset_Train_dl, 
    Dataset_Pred_dl
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_provider_dl(args, flag):
    """
    创建 torch Dataset 和 DataLoader

    Args:
        args (_type_): _description_
        flag (_type_): _description_
    """
    # 是否对时间戳进行编码
    # timeenc = 0 if args.embed != 'timeF' else 1
    # 数据集参数
    if flag in ["train", "val"]:
        shuffle_flag = False           # 是否进行 shuffle 数据
        drop_last = False             # 是否丢弃最后一个 batch
        batch_size = args.batch_size
        Data = Dataset_Train_dl          # 数据集类
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        Data = Dataset_Train_dl
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred_dl
    # 构建 Dataset 和 DataLoader
    data_creator = Data(
        args = args,
        # root_path = args.root_path,
        # data_path = args.data_path,
        flag = flag,
        # size = [args.seq_len, args.label_len, args.pred_len],
        # features = args.features,
        # target = args.target,
        # timeenc = timeenc,
        # freq = args.freq,
        # seasonal_patterns=args.seasonal_patterns,
        # scale = args.scale,
        # inverse = args.inverse,
        # cols = None,
    )
    data_x, data_y = data_creator.run()
    # data set
    data_set = TensorDataset(
        torch.from_numpy(data_x).to(torch.float32),
        torch.from_numpy(data_y).to(torch.float32)
    )
    # data loader
    data_loader = DataLoader(
        data_set, 
        batch_size = batch_size, 
        shuffle = shuffle_flag,
        num_workers = args.num_workers,
        drop_last = drop_last
    )
    
    return data_set, data_loader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
