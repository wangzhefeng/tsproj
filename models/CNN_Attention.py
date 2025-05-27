# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CNN_Attention.py
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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, feature_size, seq_len, out_channels, num_heads, output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = 14
        # 一维卷积层
        self.conv1d_1 = nn.Conv1d(
            in_channels = feature_size, 
            out_channels = out_channels[0], 
            kernel_size = 3,
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels = out_channels[0],
            out_channels = out_channels[1],
            kernel_size = 3,
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels = out_channels[1],
            out_channels = out_channels[2],
            kernel_size = 3,
        )
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim = self.hidden_size,
            num_heads = num_heads,
            batch_first = True,
            dropout = 0.8
        )
        # 输出层
        self.linear1 = nn.Linear(
            in_features = self.hidden_size * out_channels[2],
            out_features = 256,
        )
        self.linear2 = nn.Linear(256, output_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden = None):
        x = x.transpose(1, 2)  # (batch_size, feature_size, seq_len[32, 1, 20])
        # conv1d
        x = self.conv1d_1(x)  # (32, 10, 18)
        x = self.relu(x)
        x = self.conv1d_2(x)  # (32, 20, 16)
        x = self.relu(x)
        x = self.conv1d_3(x)  # (32, 30, 14)
        x = self.relu(x)
        # attention
        attn_output, attn_output_weights = self.attention(x, x, x)
        # 展开
        x = attn_output.flatten(start_dim = 1)  # (32, 420)
        # 全连接层
        x = self.linear1(x)  # (32, 256)
        x = self.relu(x)
        # output
        output = self.linear2(x)  # (32, output_size)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
