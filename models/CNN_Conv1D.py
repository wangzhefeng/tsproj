# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CNN1D.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052816
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
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Config:
    data_path = "data/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    out_channels = [10, 20, 30]  # 卷积输出通道
    # num_layers = 2  # 网络的层数
    # hidden_size = 256  # 网络隐藏层大小
    output_size = 1  # 预测未来 n 个时刻数据
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "CNN-Conv1d"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"

config = Config()


class Model(nn.Module):

    def __init__(self, feature_size, out_channels, output_size) -> None:
        super(Model, self).__init__()
        # 一维卷积
        self.conv1d_1 = nn.Conv1d(in_channels = feature_size, out_channels = out_channels[0], kernel_size = 3)
        self.conv1d_2 = nn.Conv1d(in_channels = out_channels[0], out_channels = out_channels[1], kernel_size = 3)
        self.conv1d_3 = nn.Conv1d(in_channels = out_channels[1], out_channels = out_channels[2], kernel_size = 3)
        # 输出层
        self.linear1 = nn.Linear(in_features = out_channels[2] * 14, out_features = 128)
        self.linear2 = nn.Linear(in_features = 128, out_features = output_size)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, feature_size, timestep[32, 1, 20])

        x = self.conv1d_1(x)  # [32, 10, 18]
        x = self.relu(x)

        x = self.conv1d_2(x)  # [32, 20, 16]
        x = self.relu(x)

        x = self.conv1d_3(x)  # [32, 30, 14]
        x = self.relu(x)

        x = x.flatten(start_dim = 1)  # [32, 420]

        x = self.linear1(x)
        x = self.relu(x)

        output = self.linear2(x)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
