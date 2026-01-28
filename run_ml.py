# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lgbm_power_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse

from utils.log_util import logger


def args_parse():
    parser = argparse.ArgumentParser(description='Machine Learning Time Series Forecasting')
    # basic config
    parser.add_argument('--des', type=str, default='TimeSeries Forecasting Exp', help='exp description')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecasting', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=0, help='Whether to conduct training')
    parser.add_argument('--is_testing', type=int, required=True, default=0, help='Whether to conduct testing')
    parser.add_argument('--testing_step', type=int, default=1, help="Test step")
    parser.add_argument('--is_forecasting', type=int, required=True, default=0, help='Whether to conduct forecasting')
    parser.add_argument('--model', type=str, required=True, default='Transformer', help='model name, options: [XGBoost, LightGBM]')
    # data loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, required=True, default='OT', help='target feature in S or MS task')
    parser.add_argument('--time', type=str, required=True, default='time', help='time feature in S or MS task')
    parser.add_argument('--freq', type=str, required=True, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument("--step_size", type=int, default=1, help="RNNs data window step")
    parser.add_argument('--train_ratio', type=float, required=True, default=0.7, help='train dataset ratio')
    parser.add_argument('--test_ratio', type=float, required=True, default=0.2, help='test dataset ratio')
    parser.add_argument('--scale', type=int, default=0, help = 'data transform')
    parser.add_argument('--inverse', type=int, default=0, help='inverse output data')
    # output dirs
    parser.add_argument('--checkpoints', type=str, default='./saved_results/pretrained_models/', help='location of model models')
    parser.add_argument('--test_results', type=str, default='./saved_results/test_results/', help='location of model models')
    parser.add_argument('--pred_results', type=str, default='./saved_results/predict_results/', help='location of model models') 
    # forecasting task
    parser.add_argument('--hist_len', type=int, required=True, default=72, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # model define
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--patience', type = int, default=15, help = 'early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # metrics (dtw)
    parser.add_argument('--use_dtw', type=int, default=0, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    # GPU
    parser.add_argument('--use_gpu', type=int, default=0, help='use gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help = 'use multiple gpus')
    parser.add_argument('--devices', type=str, default="0,1,2,3,4,5,6,7,8", help='device ids of multile gpus')
    
    # 命令行参数解析
    args = parser.parse_args()

    return args


def run(args):
    pass




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed_ml(seed = 2025)
    # 参数解析
    args = args_parse()
    print_args_ts(args)
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
