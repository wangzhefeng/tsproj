# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TCN.py
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

from loguru import logger
import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Config:
    data_path = "data/{}.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_channels = [32, 64, 128, 256]  # 卷积通道数
    kernel_size = 3  # 卷积核大小
    dropout = 0.2  # 丢弃率
    # num_layers = 2  # 网络的层数
    # hidden_size = 256  # 网络隐藏层大小
    output_size = 1  # 预测未来 n 个时刻数据
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "TCN"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"

config = Config()


class Model(nn.Module):
    """
    TCN(Temporal Convolutional Network)
    模型介绍：
        一种基于卷积神经网路的时间序列模型。
        它通过一系列的一维卷积层对输入序列进行特征提取，
        然后将提取到的特征输入到一个全连接层中进行预测
    预测方式：
        - 多变量预测
        - 单步、多步预测
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout) -> None:
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        # 卷积层
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(input_size, num_channels[0], kernel_size)
        )
        for i in range(1, len(num_channels)):
            self.layers.append(
                nn.Conv1d(num_channels[i - 1], num_channels[i], kernel_size)
            )
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # 全连接层
        self.linear = nn.Linear(in_features = num_channels[-1], out_features = output_size)

    def forward(self, x):
        # 将数据维度从 (batch_size, seq_len, input_size) 变为 (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        # 通过卷积层和 Dropout 层进行特征提取
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        # 将卷积层的输出求平均并输入全连接层得到最终输出
        x = x.mean(dim = 2)
        output = self.linear(x)
        return output


model = Model(
    config.feature_size, 
    config.output_size, 
    config.num_channels, 
    config.kernel_size, 
    config.dropout,
)
logger.info(model)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
