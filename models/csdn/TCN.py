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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
