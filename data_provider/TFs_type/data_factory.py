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

from torch.utils.data import DataLoader

from data_provider.TFs_type.data_loader import (
    Dataset_Train, Dataset_Pred,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_provider(args, flag):
    """
    数据集构造
    """
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != "timeF" else 1
    # 区别在 test/pred 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if flag.lower() in ["test", "pred"] else True
    # 是否丢弃最后一个 batch
    drop_last = False
    # 数据集参数
    if args.data == "m4":
        from data_provider.TFs_type.data_loader_m4 import Dataset_M4
        if flag in ["train", "valid"]:
            batch_size = args.batch_size
            Data = Dataset_M4
        elif flag in "test":
            batch_size = 1
            Data = Dataset_M4
        elif flag == "pred":
            batch_size = 1
            Data = Dataset_M4
    else:
        if flag in ["train", "valid"]:
            batch_size = args.batch_size
            Data = Dataset_Train
        elif flag in "test":
            batch_size = 1
            Data = Dataset_Train
        elif flag == "pred":
            batch_size = 1
            Data = Dataset_Pred
    # 构建 Dataset 和 DataLoader
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        logger.info(f"{flag}: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path = args.root_path,
            flag = flag,
        )
        logger.info(f"{flag}: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == "m4":
            drop_last = False
        
        data_set = Data(
            args = args,
            root_path = args.root_path,
            data_path = args.data_path,
            flag = flag,
            size = [args.seq_len, args.label_len, args.pred_len],
            features = args.features,
            target = args.target,
            time = args.time,
            timeenc = timeenc,
            freq = args.freq,
            seasonal_patterns = args.seasonal_patterns,
            scale = args.scale,
            inverse = args.inverse,
            testing_step = args.testing_step,
        )
        logger.info(f"{flag}: {len(data_set)}")
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
    from utils.args_tools import DotDict
    args = {
        "task_name": "short_term_forecast",
        "flag": "train",
        "embed": "timeF",
        "root_path": "./dataset/m4",
        "data_path": "ETTh1.csv",
        "seasonal_patterns": "Monthly",
        "model_id": "m4_Monthly",
        "data": "m4",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seq_len": 6,
        "pred_len": 3,
        "label_len": 3,
        "train_ratio": 0.7,
        "test_ratio": 0.2,
        "batch_size": 1,
        "scale": False,
        "inverse": False,
        "num_workers": 0,
        "testing_step": 1,
    }
    args = DotDict(args)

    data_set, data_loader = data_provider(args, flag="train")
    # data_set, data_loader = data_provider(args, flag="valid")
    # data_set, data_loader = data_provider(args, flag="test")
    # data_set, data_loader = data_provider(args, flag="pred")

if __name__ == "__main__":
    main()
