# -*- coding: utf-8 -*-

# ***************************************************
# * File        : get_args.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-24
# * Version     : 1.0.012416
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_args_script_A3_2():
    # args
    from utils.tools import dotdict
    args = dotdict()
    # 任务类型参数
    args.is_training = True
    args.is_predicting = True
    # 数据参数
    args.root_path = "./dataset/electricity/"
    args.data_path = "all_df.csv"
    args.rolling_data_path = "ETTh1-Test.csv"
    args.target = "204_load"
    args.freq = "h"
    args.embed = "timeF"
    args.seq_len = 72
    args.label_len = 12
    args.pred_len = 24
    # 模型定义参数
    args.model = "Transformer"
    args.rollingforecast = True
    args.embed_type = 0
    args.d_model = 64
    args.enc_in = 10793
    args.dec_in = 10793
    args.e_layers = 4
    args.d_layers = 4
    args.n_heads = 1
    args.d_ff = 2048
    args.activation = "gelu"
    args.c_out = 1
    args.dropout = 0.05
    args.rev = True
    args.output_attention = False
    args.padding = 0
    args.loss = "MSE"
    # 模型训练参数
    args.features = "MS"
    args.iters = 10
    args.train_epochs = 10
    args.batch_size = 1
    args.learning_rate = 1e-5
    args.patience = 7
    args.lradj = "type1"
    args.checkpoints = "./saved_results/pretrained_models/"
    args.test_results = "./saved_results/test_results/"
    args.predict_results = './saved_results/predict_results/'
    args.show_results = True
    args.inverse = False
    args.scale = False
    args.use_gpu = False
    args.use_multi_gpu = False
    args.device_id = 0
    args.num_workers = 0
    logger.info(f"Args in experiment: \n{args}")
    
    return args


def get_args_script_A3_3():
    # args
    from utils.tools import dotdict
    args = dotdict()
    # 任务类型参数
    args.is_training = True
    args.is_predicting = True
    # 数据参数
    args.root_path = "./dataset/electricity/"
    args.data_path = "all_df_A3_301_302.csv"
    args.rolling_data_path = "ETTh1-Test.csv"
    args.target = "302_load"
    args.freq = "h"
    args.embed = "timeF"
    args.seq_len = 72
    args.label_len = 12
    args.pred_len = 24
    # 模型定义参数
    args.model = "Transformer"
    args.rollingforecast = True
    args.embed_type = 0
    args.d_model = 64
    args.dropout = 0.05
    args.enc_in = 6946
    args.dec_in = 6946
    args.e_layers = 4
    args.d_layers = 4
    args.n_heads = 1
    args.d_ff = 2048
    args.activation = "gelu"
    args.c_out = 1
    args.rev = True
    args.output_attention = False
    args.padding = 0
    args.loss = "MSE"
    # 模型训练参数
    args.features = "MS"
    args.iters = 10
    args.train_epochs = 10
    args.batch_size = 1
    args.learning_rate = 1e-5
    args.patience = 7
    args.lradj = "type1"
    args.checkpoints = "./saved_results/pretrained_models/"
    args.test_results = "./saved_results/test_results/"
    args.predict_results = './saved_results/predict_results/'
    args.show_results = True
    args.inverse = True
    args.scale = True
    args.use_gpu = False
    args.use_multi_gpu = False
    args.device_id = 0
    args.num_workers = 0
    logger.info(f"Args in experiment: \n{args}")
    
    return args


def get_args_script_ETTh():
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
    args.embed = "timeF"  # timeF,fixed,learned
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    # 模型定义参数
    args.model = "Transformer"
    args.embed_type = 0
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 1  # TODO the number of features or channels
    args.rev = True
    args.d_model = 64
    args.dropout = 0.05
    args.e_layers = 2
    args.d_layers = 2
    args.n_heads = 8
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
    args.predict_results = './saved_results/predict_results/'
    args.show_results = True
    args.inverse = True
    args.scale = True
    args.rollingforecast = False
    args.use_gpu = False
    args.use_multi_gpu = False
    args.device_id = 0
    args.num_workers = 0
    logger.info(f"Args in experiment: \n{args}")
    
    return args






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
