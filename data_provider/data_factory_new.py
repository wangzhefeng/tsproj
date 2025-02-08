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
    Dataset_Custom,
    Dataset_Test,
    Dataset_Pred,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_provider(args, flag, pre_data):
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
        Data = Dataset_Custom
    elif flag == 'val':
        batch_size = args.batch_size
        Data = Dataset_Custom
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


def get_args_script():
    # args
    from utils.tools import dotdict
    args = dotdict()
    # 任务类型参数
    args.is_training = False
    args.is_predicting = False
    # 数据参数
    args.root_path = "./dataset/tf_data/"
    args.data_path = "ETTh1.csv"
    args.rolling_data_path = "ETTh1-Test.csv"
    args.target = "OT"
    args.freq = "h"
    args.embed = "timeF"  # timeF, fixed
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    # 模型定义参数
    args.model = "Transformer"
    args.embed_type = 0
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7                  # TODO the number of features or channels
    args.rev = True
    args.d_model = 64
    args.dropout = 0.05
    args.e_layers = 2
    args.d_layers = 1
    args.n_heads = 1
    args.d_ff = 2048
    args.activation = "gelu"
    args.output_attention = False
    args.padding = 0
    args.loss = "MSE"
    # 模型训练参数
    args.features = "MS"
    args.iters = 1
    args.train_epochs = 1
    args.batch_size = 1
    args.learning_rate = 1e-4
    args.patience = 7
    args.lradj = "type1"
    args.checkpoints = "./saved_results/pretrained_models/"
    args.test_results = "./saved_results/test_results/"
    args.predict_results = f'./saved_results/predict_results/'
    args.show_results = True
    args.inverse = False
    args.scale = False
    args.rollingforecast = True
    args.use_gpu = False
    args.use_multi_gpu = False
    args.device_id = 0
    args.num_workers = 0
    logger.info(f"Args in experiment: \n{args}")
    
    return args




# 测试代码 main 函数
def main():
    from exp.exp_forecast_dl import Exp_Forecast
    # params
    args = get_args_script()
    
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
