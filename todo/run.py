# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-31
# * Version     : 0.1.053120
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
import argparse
import random

import numpy as np
import torch
from loguru import logger

from experiments.exp_anomaly_detection import Exp_Anomaly_Detection
from experiments.exp_classification import Exp_Classification
from experiments.exp_imputation import Exp_Imputation
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_short_term_forecasting import Exp_Short_Term_Forecast

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def set_seed():
    """
    设置可重复随机数
    """
    fix_seed = 2023
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
        

def args_define():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description = "TimeseriesForecasting")
    # ------------------------------
    # add argument
    # ------------------------------
    # basic config
    parser.add_argument('--task_name', type = str, required = True, default = 'long_term_forecast', help = 'task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type = int, required = True, default = 1, help = 'status')
    parser.add_argument('--model_id', type = str, required = True, default = 'test', help = 'model id')
    parser.add_argument('--model', type = str, required = True, default = 'Autoformer', help = 'model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--data', type = str, required = True, default = 'ETTm1', help = 'dataset type')
    parser.add_argument('--root_path', type = str, default = 'data/', help = 'root path of the data file')
    parser.add_argument('--data_path', type = str, default = 'ETTh1.csv', help = 'data file')
    parser.add_argument('--features', type = str, default = 'M', help = 'forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type = str, default = 'OT', help = 'target feature in S or MS task')
    parser.add_argument('--freq', type = str, default = 'h', help = 'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type = str, default = 'checkpoints/', help = 'location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type = int, default = 96, help = 'input sequence length')
    parser.add_argument('--label_len', type = int, default = 48, help = 'start token length')
    parser.add_argument('--pred_len', type = int, default = 96, help = 'prediction sequence length')
    parser.add_argument('--seasonal_patterns', type = str, default = 'Monthly', help = 'subset for M4')
    # TODO inputation task
    # parser.add_argument('--mask_rate', type = float, default = 0.25, help = 'mask ratio')
    # TODO anomaly detection task
    # parser.add_argument('--anomaly_ratio', type = float, default = 0.25, help = 'prior anomaly ratio (%)')
    # model define
    parser.add_argument('--top_k', type = int, default = 5, help = 'for TimesBlock')
    parser.add_argument('--num_kernels', type = int, default = 6, help = 'for Inception')
    parser.add_argument('--enc_in', type = int, default = 7, help = 'encoder input size')
    parser.add_argument('--dec_in', type = int, default = 7, help = 'decoder input size')
    parser.add_argument('--c_out', type = int, default = 7, help = 'output size')
    parser.add_argument('--d_model', type = int, default = 512, help = 'dimension of model')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'num of heads')
    parser.add_argument('--e_layers', type = int, default = 2, help = 'num of encoder layers')
    parser.add_argument('--d_layers', type = int, default = 1, help = 'num of decoder layers')
    parser.add_argument('--d_ff', type = int, default = 2048, help = 'dimension of fcn')
    parser.add_argument('--moving_avg', type = int, default = 25, help = 'window size of moving average')
    parser.add_argument('--factor', type = int, default = 1, help = 'attn factor')
    parser.add_argument('--distil', action = 'store_false', default = True, help = 'whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--dropout', type = float, default = 0.1, help = 'dropout')
    parser.add_argument('--embed', type = str, default = 'timeF', help = 'time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type = str, default = 'gelu', help = 'activation')
    parser.add_argument('--output_attention', action = 'store_true', help = 'whether to output attention in ecoder')
    # optimization
    parser.add_argument('--num_workers', type = int, default = 10, help = 'data loader num workers')
    parser.add_argument('--itr', type = int, default = 1, help = 'experiments times')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size of train input data')
    parser.add_argument('--patience', type = int, default = 3, help = 'early stopping patience')
    parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'optimizer learning rate')
    parser.add_argument('--des', type = str, default = 'test', help = 'exp description')
    parser.add_argument('--loss', type = str, default = 'MSE', help = 'loss function')
    parser.add_argument('--lradj', type = str, default = 'type1', help = 'adjust learning rate')
    parser.add_argument('--use_amp', action = 'store_true', default = False, help = 'use automatic mixed precision training')
    # GPU
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu')
    parser.add_argument('--use_multi_gpu', action = 'store_true', default = False, help = 'use multiple gpus')
    parser.add_argument('--devices', type = str, default = '0,1,2,3', help = 'device ids of multile gpus')
    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type = int, nargs='+', default = [128, 128], help = 'hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type = int, default = 2, help = 'number of hidden layers in projector')

    return parser


def args_parse(parser):
    # ------------------------------
    # 参数解析
    # ------------------------------
    # arguments
    args = parser.parse_args()
    # device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # device-gpu
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    logger.info(f"Args in experiment: \n{args}")

    return args
    

def args_usage(args):
    # ------------------------------
    # 模型任务
    # ------------------------------
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    # elif args.task_name == 'imputation':
    #     Exp = Exp_Imputation
    # elif args.task_name == 'anomaly_detection':
    #     Exp = Exp_Anomaly_Detection
    # elif args.task_name == 'classification':
    #     Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast
    # ------------------------------
    # 模型训练
    # ------------------------------
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features} \
                        _sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model} \
                        _nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff} \
                        _fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"
            # set experiments
            exp = Exp(args)
            # model training
            logger.info(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            # model testing
            logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            # empty cache
            torch.cuda.empty_cache()
    else:
        ii = 0
        # setting record of experiments
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features} \
                    _sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model} \
                    _nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff} \
                    _fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"
        # set experiments
        exp = Exp(args)
        logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test = 1)
        torch.cuda.empty_cache()




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed()
    # 参数定义
    parser = args_define()
    # 参数解析
    args = args_parse(parser)
    # 参数使用
    args_usage(args) 

if __name__ == "__main__":
    main()
