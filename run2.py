# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run2.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-21
# * Version     : 0.1.082121
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import random
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 设置随机数
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True


# 设备配置
use_gpu = True if torch.cuda.is_available() else False
use_multi_gpu = False
devices = "0,1,2,3"
device_ids = [int(id_) for id_ in devices.replace(' ', '').split(',')]
if use_gpu and use_multi_gpu:
    gpu = devices
elif use_gpu and not use_multi_gpu:
    gpu = device_ids[0]


class Config:
    # 设备参数
    use_gpu = use_gpu
    use_multi_gpu = use_multi_gpu
    devices = devices
    device_ids = device_ids
    gpu = gpu
    num_workers = 1
    use_amp = False
    # 任务类型参数
    task_name = "long_term_forecast"
    is_training = True
    # 数据参数
    root_path = "dataset/ETT-small/"
    data_path = "ETTm1.csv"
    data = "ETTm1"
    seq_len = 96
    label_len = 48
    pred_len = 96
    features = "M"
    target = "OT"
    freq = "t"
    seasonal_patterns = "Monthly"
    scale = True
    embed = "fixed"
    augmentation_ratio = 0
    # 模型定义参数
    model_id = "ETTm1_96_96"
    model = "TimesNet"
    num_kernels = 6
    e_layers = 2
    # d_layers = 1
    # factor = 3
    enc_in = 7
    # dec_in = 7
    c_out = 7
    # des = "Exp"
    dropout = 0.1
    d_model = 64
    d_ff = 64
    top_k = 5 
    # 模型训练参数
    iters = 1
    train_epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    patience = 3
    checkpoints = "checkpoints/"
    test_results = "test_results/"
    results = "results/"
    # 其他
    # n_heads = 8
    # expand = 2
    # d_conv = 4
    # distil = True

args = Config()


# 模型任务
if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
else:
    Exp = Exp_Long_Term_Forecast


# 模型训练、验证
if args.is_training:
    for ii in range(args.iters):
        # 迭代轮数
        args.ii = ii
        # 创建任务实例
        exp = Exp(args)
        setting = f"{args.task_name}_{args.model_id}"
        # 模型训练、验证
        exp.train(setting)
        # 模型验证
        exp.test(setting)
        # 清除 CUDA 缓存
        torch.cuda.empty_cache()
else:
    # 创建任务实例
    exp = Exp(args)
    setting = f"{args.task_name}_{args.model_id}"
    # 模型测试
    exp.test(setting, test = 1)
    # 清除 CUDA 缓存
    torch.cuda.empty_cache()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
