# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM_CNN.py
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


class Model(nn.Module):

    def __init__(self, 
                 feature_size, 
                 timestep, 
                 hidden_size, 
                 num_layers, 
                 out_channels, 
                 output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = nn.LSTM(
            input_size = feature_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels = timestep,
            out_channels = out_channels,
            kernel_size = 3,
        )
        # 输出层
        self.linear1 = nn.Linear(in_features = 50 * 254, out_features = 256)
        self.linear2 = nn.Linear(in_features = 256, out_features = output_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # (batch_size, timestep, hidden_size)
        # 卷积
        output = self.conv1d(output)
        # 展开
        output = output.flatten(output)
        # 全连接层
        output = self.linear1(output)
        output = self.relu(output)

        output = self.linear2(output)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
