# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM_Attention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052816
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, 
                 feature_size, 
                 seq_len, 
                 hidden_size, 
                 num_layers, 
                 num_heads, 
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
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_size, 
            num_heads = num_heads, 
            batch_first = True,
            dropout = 0.8,
        )
        # 输出层
        self.linear1 = nn.Linear(
            in_features = hidden_size * seq_len, 
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
        # 初始隐藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # output.shape:[32, 20, 64]
        # Attention
        attn_output, attn_output_weights = self.attention(output, output, output)
        # 展开
        output = attn_output.flatten(start_dim = 1)  # [32, 1280]
        # 全连接层
        output = self.linear1(output)  # [32, 256]
        output = self.relu(output)

        output = self.linear2(output)  # [32, output_size]
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
