# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CNN_LSTM_Attention.py
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
                 seq_len, 
                 hidden_size, 
                 num_layers, 
                 out_channels, 
                 num_heads, 
                 output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels = feature_size,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1,
        )
        # LSTM 层
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = out_channels, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True,
        )
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim = self.hidden_size,
            num_heads = num_heads,
            batch_first = True,
            dropout = 0.8,
        )
        # 输出层
        self.linear1 = nn.Linear(
            in_features = seq_len * hidden_size, 
            out_features = 256
        )
        self.linear2 = nn.Linear(
            in_features = 256,
            out_features = output_size,
        )
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        x = x.transposep(1, 2)  # (batch_size, feature_size, seq_len[32, 1, 20])
        # 卷积
        output = self.conv1d(x)
        output = output.transpose(1, 2)  # (batch_size, feature_size, seq_len[32, 1, 20])
        # 初始化隐藏层
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(output, (h_0, c_0))  # (batch_size, seq_len, hidden_size)
        # 注意力机制
        attn_output, attn_output_weights = self.attention(output, output, output)
        # 展开
        output = output.flatten(start_dim = 1)
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
