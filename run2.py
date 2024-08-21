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
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import random
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# 设置随机数
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

# 设备配置
use_gpu = True if torch.cuda.is_available() else False
use_multi_gpu = False
devices = "0,1,2,3"
device_ids = [int(id_) for id_ in devices.replace(' ', '').split(',')]
if use_gpu and use_multi_gpu:
    gpu = devices
elif use_gpu and not use_multi_gpu:
    gpu = device_ids[0]
print(f"use_gpu: {use_gpu}")
print(f"use_multi_gpu: {use_multi_gpu}")
print(f"gpu: {gpu}") 
print("-" * 80)


class Config:
    task_name = "long_term_forecast"
    is_training = True
    
    root_path = "dataset/ETT-small/"
    data_path = "ETTm1.csv"
    data = "ETTm1"
    freq = "m"
    
    model_id = "ETTm1_96_96"
    model = "TimesNet"
    features = "M"
    target = "OT"
    
    seq_len = 96
    label_len = 48
    pred_len = 96
    d_model = 64
    batch_size = 32
    dropout = 0.1
    top_k = 5
    
    e_layers = 2
    d_layers = 1
    factor = 3
    enc_in = 7
    dec_in = 7
    num_kernels = 6
    c_out = 7
    d_ff = 64
    des = "Exp"
    embed = "fixed"

    use_gpu = use_gpu
    use_multi_gpu = use_multi_gpu
    devices = devices
    device_ids = device_ids
    gpu = gpu 


# 模型训练、验证
itr = 1
for ii in range(itr):
    args = Config()
    args.ii = ii
    args.n_heads = 8
    args.expand = 2
    args.d_conv = 4
    args.distil = True 
    args.seasonal_patterns = "Monthly"
    args.augmentation_ratio = 0
    args.num_workers = 10
    args.checkpoints = "checkpoints/"
    args.patience = 3
    args.learning_rate = 0.0001
    args.use_amp = False
    args.train_epochs = 10
    # 创建任务实例
    exp = Exp_Long_Term_Forecast(args)
    setting = f"{args.task_name}_{args.model_id}"
    # # 模型训练
    exp.train(setting)
    # # 模型测试
    # exp.test(setting_train)
    # # 清除 CUDA 缓存
    # torch.cuda.empty_cache()


# 模型测试
# ii = 0
# args = {
#         "task_name": "long_term_forecast",
#         "is_training": True,
#         "root_path": "dataset/ETT-small/",
#         "data_path": "ETTm1.csv",
#         "model_id": "ETTm1_96_96",
#         "model": "TimesNet",
#         "data": "ETTm1",
#         "features": "M",
#         "seq_len": 96,
#         "label_len": 48,
#         "pred_len": 96,
#         "e_layers": 2,
#         "d_layers": 1,
#         "factor": 3,
#         "enc_in": 7,
#         "dec_in": 7,
#         "c_out": 7,
#         "des": "Exp",
#         "d_model": 64,
#         "d_ff": 64,
#         "top_k": 5,
#         "ii": ii,
#     }
# setting = f"{args.task_name}_{args.model_id}"
# exp = Exp_Long_Term_Forecast(args)
# exp.test(setting, test = 1)
# # 清除 CUDA 缓存
# torch.cuda.empty_cache()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
