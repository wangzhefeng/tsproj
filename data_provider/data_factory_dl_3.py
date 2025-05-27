# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory_dl_3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-24
# * Version     : 1.0.052418
# * Description : https://blog.csdn.net/java1314777/article/details/134407174
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader

from data_provider.data_loader_dl_3 import TimeSeriesDataset

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_provider(args, flag):
    """
    数据集构造
    """
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != "timeF" else 1
    # 区别在 test/pred 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if flag in ["test", "pred"] else True
    # 是否丢弃最后一个 batch
    drop_last = False# if flag in ["pred"] else True
    # 数据集参数
    if flag in ["train", "vali"]:
        batch_size = args.batch_size
        Data = TimeSeriesDataset
    elif flag in "test":
        batch_size = 1
        Data = TimeSeriesDataset
    elif flag == "pred":
        batch_size = 1
        Data = TimeSeriesDataset
    # 构建 Dataset 和 DataLoader
    data_set = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        target = args.target,
        features = args.features,
        window_len = args.window_len,
        pred_len = args.pred_len,
        step_size = args.step_size,
        scale = args.scale,
        flag = flag
    )
    data_loader = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        num_workers=args.num_workers,
        drop_last = drop_last,
    )
    
    return data_set, data_loader




# 测试代码 main 函数
def main():
    from utils.args_tools import DotDict
    args = {
        "embed": "timeF",
        "batch_size": 1,
        "root_path": ".\\dataset\\ETT-small",
        "data_path": "ETTh1.csv",
        "target": "OT",
        "features": "S",
        "window_len": 6,
        "pred_len": 1,
        "step_size": 1,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "scale": True,
        "num_workers": 0
    }
    args = DotDict(args)
    # data
    data_set, data_loader = data_provider(args, flag="train")
    data_set, data_loader = data_provider(args, flag="test")

if __name__ == "__main__":
    main()
