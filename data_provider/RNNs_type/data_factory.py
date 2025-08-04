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
# ***************************************************

__all__ = [
    "data_provider"
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.data import DataLoader

from data_provider.RNNs_type.data_loader import (
    Dataset_Train,
    Dataset_Pred,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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
    data_set = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        target = args.target,
        time = args.time,
        freq = args.freq,
        features = args.features,
        seq_len = args.seq_len,
        pred_len = args.pred_len,
        step_size = args.step_size,
        scale = args.scale,
        flag = flag
    )
    data_loader = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        sampler=None,
        drop_last = drop_last,
        num_workers=args.num_workers,
    )
    
    return data_set, data_loader




# 测试代码 main 函数
def main():    
    # command arguments
    args = { 
        # data 
        # ----------------------------
        "root_path": "./dataset/ETT-small",  # 数据集目录
        "data_path": "ETTh1.csv",  # 数据文件名
        "target": "OT",  # 数据目标特征
        "time": "date",  # 数据时间列名
        "freq": "h",  # 数据频率
        "seq_len": 6,  # 窗口大小(历史)
        "pred_len": 3,  # 预测长度
        "step_size": 1,  # 滑窗步长
        "batch_size": 1,
        "train_ratio": 0.7,
        "test_ratio": 0.2, 
        "embed": "timeF",
        "scale": True,
        "num_workers": 0,
        # task
        # ----------------------------
        "features": "MS",
        "feature_size": 7,  # 特征个数(除了时间特征)
        "hidden_size": 128,
        "num_layers": 2,
        "rolling_predict": True,  # 是否进行滚动预测功能
        "rolling_data_path": "ETTh1Test.csv"  # 滚动数据集的数据
    }
    from utils.args_tools import DotDict
    args = DotDict(args)
    
    # data
    data_set, data_loader = data_provider(args, flag="train")
    # data_set, data_loader = data_provider(args, flag="valid")
    # data_set, data_loader = data_provider(args, flag="test")

    # data test
    from utils.log_util import logger
    for input_seq, label_seq in data_loader:
        logger.info(f"input_seq: \n{input_seq} \ninput_seq.shape: {input_seq.shape}")
        logger.info(f"label_seq: \n{label_seq} \nlabel_seq.shape: {label_seq.shape}")
        break

if __name__ == "__main__":
    main()
