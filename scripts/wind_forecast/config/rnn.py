# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_rnn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-06-11
# * Version     : 0.1.061115
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128619443
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Config:
    data_path = "dataset/wind_dataset.csv"
    split_ratio = 0.8
    seq_len = 1  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # 网络的层数
    hidden_size = 256  # 网络隐藏层大小
    output_size = 1  # 预测未来 n 个时刻数据
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "RNN"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()