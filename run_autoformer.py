# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-08
# * Version     : 1.0.010821
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import argparse

import torch

from exp.exp_forecast_ltf import Exp_Forecast
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def args_parse():
    parser = argparse.ArgumentParser(description='Transformer Multivariate Time Series Forecasting')
    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='Whether to conduct training')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_name', type=str, required=True, default='Transformer', help='Model name')
    parser.add_argument('--do_forecasting', type=int, required=True, default=0, help='Whether to conduct forecasting')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecasting', help="long term forecasting")
    # data loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, required=True, default='OT', help='target feature in S or MS task')    
    parser.add_argument('--freq', type=str, required=True, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, required=True, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    # forecasting task
    parser.add_argument('--seq_len', type=int, required=True, default=72, help='input sequence length')
    parser.add_argument('--label_len', type=int, required=True, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, required=True, default=24, help='prediction sequence length')
    # model config
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--embed_type', type=int, default=0, help='0: value embedding + temporal embedding + positional embedding 1: value embedding + positional embedding')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--rev', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--output_attention', type=int, default=0, help='whether to output attention in ecoder')
    parser.add_argument('--padding', type=int, default=0, help='padding')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--use_dtw', type=int, default=0, help='loss function')
    # model training config
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--iters', type=int, default=10, help='train iters')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate')
    parser.add_argument('--patience', type = int, default = 3, help = 'early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # path
    parser.add_argument('--checkpoints', type=str, default='./saved_results/pretrained_models/', help='location of model models')
    parser.add_argument('--test_results', type=str, default='./saved_results/test_results/', help='location of model models')
    parser.add_argument('--predict_results', type=str, default='./saved_results/predict_results/', help='location of model models')
    parser.add_argument('--show_results', type=int, default=1, help='Whether show forecast and real results graph')
    # data transform
    parser.add_argument('--scale', type=int, default = 1, help = 'data transform')
    parser.add_argument('--inverse', type=int, default = 1, help = 'inverse output data')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default="0", help='gpu id')
    parser.add_argument('--use_multi_gpu', type=bool, default = False, help = 'use multiple gpus')
    parser.add_argument('--devices', type=str, default="0,1,2,3,4,5,6,7,8", help='device ids of multile gpus')
    # 命令行参数解析 
    args = parser.parse_args()
    
    # device
    args.use_gpu = True if torch.cuda.is_available() else False
    # device-gpu
    args.devices = args.devices.replace(" ", "")
    if args.use_gpu and args.use_multi_gpu:
        args.gpu = args.devices
    elif args.use_gpu and not args.use_multi_gpu:
        args.gpu = [int(id_) for id_ in args.devices.split(",")][0]
    
    print(f"Args in experiment: \n{args}")

    return args


def run(args):
    # params
    # args = get_args_script_A3_3()
     # setting record of experiments
    setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_".format(
        args.model_id,
        args.model_name,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, 
    )
    # 模型训练
    if args.is_training:
        for ii in range(args.iters):
            # setting record of experiments
            setting = setting + str(ii)
            # 实例化模型
            exp = Exp_Forecast(args)
            # 模型训练
            logger.info(f">>>>>>>start training: iter-{ii}: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting, itr = ii)
            # 模型测试
            logger.info(f">>>>>>>start testing: iter-{ii}: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.test(setting, test = 0)
            # 模型预测
            if args.do_forecasting:
                logger.info(f">>>>>>>start forecasting: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
                exp.forecast(setting, load = True)
            # empty cache
            torch.cuda.empty_cache()
    else:
        ii = 0
        # setting record of experiments
        setting = setting + str(ii)
        # 实例化模型
        exp = Exp_Forecast(args)
        # 模型测试
        logger.info(f">>>>>>>start testing: iter-{ii}: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.test(setting, test = 1)
        # empty cache
        torch.cuda.empty_cache()




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed(seed = 2023)
    # 参数解析
    args = args_parse()
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
