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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
from torch.utils.data import TensorDataset, DataLoader

from data_provider.data_loader_dl_1 import (
    Dataset_Train, 
    Dataset_Pred
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_provider(args, flag):
    """
    创建 torch Dataset 和 DataLoader
    """
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != 'timeF' else 1
    # 区别在 test/pred 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if flag in ["test", "pred"] else True
    # 是否丢弃最后一个 batch
    drop_last = False# if flag in ["pred"] else True
    # 数据集参数
    if flag in ["train", "val"]:
        batch_size = args.batch_size
        Data = Dataset_Train
    elif flag == 'test':
        batch_size = 1
        Data = Dataset_Train
    elif flag == 'pred':
        batch_size = 1
        Data = Dataset_Pred
    # 构建 Dataset 和 DataLoader
    data_creator = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        flag = flag,
        # size = [args.seq_len, args.label_len, args.pred_len],
        seq_len=args.seq_len,
        feature_size=args.feature_size,
        output_size=args.output_size,
        target = args.target,
        features = args.features,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        pred_method=args.pred_method,
        # freq = args.freq,
        # timeenc = timeenc,
        # seasonal_patterns=args.seasonal_patterns,
        scale = args.scale,
        inverse = args.inverse,
    )
    data_set = TensorDataset(
        torch.from_numpy(data_creator.data_x).to(torch.float32),
        torch.from_numpy(data_creator.data_y).to(torch.float32)
    )
    # logger.info(f"debug::data_set: \n{data_set} \nlen(data_set): {data_set.__dict__}")
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
    from utils.args_tools import DotDict
    # ------------------------------
    # recursive_multi_step
    # ------------------------------
    args_s = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "S",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 1,
        "feature_size": 1,
        "output_size": 1,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 2,
        "pred_method": "recursive_multi_step",
        "num_workers": 0,
    }
    args_ms = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "MS",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 1,
        "feature_size": 8,
        "output_size": 1,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 1,
        "pred_method": "recursive_multi_step",
        "num_workers": 0,
    }
    # ------------------------------
    # direct_multi_step
    # ------------------------------
    args_s = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "S",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 2,
        "feature_size": 1,
        "output_size": 2,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 1,
        "pred_method": "direct_multi_output",
        "num_workers": 0,
    }
    args_ms = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "MS",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 2,
        "feature_size": 8,
        "output_size": 2,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 1,
        "pred_method": "direct_multi_output",
        "num_workers": 0,
    }
    """
    # ------------------------------
    # direct_recursive_multi_step_mix
    # ------------------------------
    args_s = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "S",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 2,
        "feature_size": 1,
        "output_size": 4,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 1,
        "pred_method": "direct_recursive_multi_step_mix",
        "num_workers": 0,
    }
    args_ms = {
        "root_path": "./dataset",
        "data_path": "wind_dataset.csv",
        "features": "MS",
        "target": "WIND",
        "embed": "timeF",
        "scale": False,
        "inverse": False,
        "seq_len": 1,
        "feature_size": 8,
        "output_size": 1,
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "batch_size": 1,
        "pred_method": "direct_recursive_multi_step_mix",
        "num_workers": 0,
    }
    """
    args = DotDict(args_s)

    # data
    data_set, data_loader = data_provider(args, flag = "train")

if __name__ == "__main__":
    main()
