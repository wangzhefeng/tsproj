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
from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom, Dataset_ETT_hour,
    Dataset_ETT_minute, Dataset_M4,
    PSMSegLoader, MSLSegLoader,
    SMAPSegLoader, SMDSegLoader,
    SWATSegLoader, UEAloader
)
from data_provider.data_loader import (
    Dataset_Train,
    Dataset_Test,
    Dataset_Pred,
)
from data_provider.uea import collate_fn
# from data_provider.data_loader import Data_Loader

from utils.log_util import logger

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
    # "Data_Loader": Data_Loader,
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


def data_provider_new(args, flag, pre_data):
    """
    数据集构造

    Args:
        args (_type_): 参数
        flag (_type_): 任务标签
        pre_data (_type_): TODO

    Returns:
        _type_: _description_
    """
    # 是否对时间戳进行编码
    timeenc = 0 if args.embed != 'timeF' else 1
    # 区别在 test 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False# if (flag == 'test' or flag == 'TEST') else True
    # 是否丢弃最后一个 batch
    drop_last = True
    # 数据集参数
    if flag == "train":
        batch_size = args.batch_size
        Data = Dataset_Train
    elif flag == 'val':
        batch_size = args.batch_size
        Data = Dataset_Train
    elif flag == 'test':
        batch_size = args.batch_size
        Data = Dataset_Test
    elif flag == 'pred':
        batch_size = 1
        pre_data = pre_data
        Data = Dataset_Pred
    
    # 构建 Dataset 和 DataLoader
    # Dataset
    data_set = Data(
        root_path = args.root_path,
        data_path = args.data_path,
        pre_data = pre_data,
        flag = flag,
        size = [args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        scale = args.scale,
        timeenc = timeenc,
        freq = args.freq,
        inverse = args.inverse,
        cols = None,
    )
    logger.info(f"Task: {flag}, data_set len: {len(data_set)}")
    # DataLoader
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
    # ------------------------------
    # data_provider_new test
    # ------------------------------
    from scripts.aidc_load_forecast.get_args import get_args_script_ETTh
    from exp.todo.exp_forecast_dl import Exp_Forecast
    # params
    args = get_args_script_ETTh()
    
    # 实例化模型
    exp = Exp_Forecast(args)
    
    # data
    train_data, train_loader = exp._get_data(flag='pred', pre_data=None)
    
    train_steps = len(train_loader)
    logger.info(f"train_steps: {train_steps}")
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        # 当前 epoch 的迭代次数记录
        # iter_count += 1
        
        # 模型优化器梯度归零
        # model_optim.zero_grad()
        
        # 数据预处理
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        logger.info(f"{i}, batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape}")
        logger.info(f"{i}, batch_x: \n{batch_x}")
        logger.info(f"{i}, batch_y: \n{batch_y}")
        
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        logger.info(f"{i}, batch_x_mark.shape: {batch_x_mark.shape}, batch_y_mark.shape: {batch_y_mark.shape}")
        logger.info(f"{i}, batch_x_mark: \n{batch_x_mark}")
        logger.info(f"{i}, batch_y_mark: \n{batch_y_mark}")
        # ------------------------------
        # 前向传播
        # ------------------------------
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
        logger.info(f"{i}, dec_inp.shape: {dec_inp.shape}")
        logger.info(f"{i}, dec_inp: \n{dec_inp}")
        
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        logger.info(f"{i}, dec_inp.shape: {dec_inp.shape}")
        logger.info(f"{i}, dec_inp: \n{dec_inp}")
        
        # encoder-decoder
        # outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # logger.info(f"{i}, outputs.shape: {outputs.shape}")
        # logger.info(f"{i}, outputs: \n{outputs}")
        
        # ------------------------------
        # DataEmbedding
        # ------------------------------
        from layers.Embed import TokenEmbedding, PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

        token_embed = TokenEmbedding(c_in=7, d_model=64)
        x = token_embed(batch_x)
        print(x)
        print(x.shape)
        
        pos_embed = PositionalEmbedding(d_model=64)
        x = pos_embed(batch_x)
        print(x)
        print(x.shape)
        
        temporal_embed = TemporalEmbedding(
            d_model=64, embed_type=args.embed, freq=args.freq,
        ) if args.embed != 'timeF' else TimeFeatureEmbedding(
            d_model=64, embed_type=args.embed, freq=args.freq,
        )
        x = temporal_embed(batch_x_mark)
        print(x)
        print(x.shape)
        
        if i == 0: break

if __name__ == "__main__":
    main()
