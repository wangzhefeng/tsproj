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
    Dataset_Train,
    Dataset_Test,
    Dataset_Pred,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_provider(args, flag):
    """
    数据集构造
    """
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != 'timeF' else 1
    # 区别在 test/pred 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False# if flag in ['test', 'pred'] else True
    # 是否丢弃最后一个 batch
    drop_last = False if flag in ["pred"] else True
    # 数据集参数
    if flag in ["train", "val"]:
        batch_size = args.batch_size
        Data = Dataset_Train
    elif flag == 'test':
        batch_size = 1
        # batch_size = args.batch_size
        Data = Dataset_Test
    elif flag == 'pred':
        batch_size = 1
        Data = Dataset_Pred
    # 构建 Dataset 和 DataLoader
    data_set = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        flag = flag,
        size = [args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        timeenc = timeenc,
        freq = args.freq,
        seasonal_patterns = args.seasonal_patterns,
        scale = args.scale,
        inverse = args.inverse,
        cols = None,
    )
    # logger.info(f"{flag.capitalize()} dataset length: {len(data_set)}")
    data_loader = DataLoader(
        data_set,
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
