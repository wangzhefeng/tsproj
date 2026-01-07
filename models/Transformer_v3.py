# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Transformer.py
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
import math

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 1) -> None:
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        # 初始化 shape 为 (max_len, d_model) 的 Positional Encoding
        pe = torch.zeros(max_len, d_model)
        # 初始化一个 tensor: [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是 sin 和 cos 括号中的内容，通过 e 和 ln 进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 计算 PE(pos, 2i)
        pe[:, 0:2] = torch.sin(position * div_term)
        # 计算 PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了计算方面，在最外面再 unsqueeze 出一个 batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存模型的时候将其保存下来，这个时候就可以用 regisiter_buffer
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # 将 x 和 positional encoding 相加
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        output = self.dropput(x)
        return output


class Model(nn.Module):

    def __init__(self, 
                 hidden_size, 
                 num_layers, 
                 feature_size, 
                 output_size, 
                 feedforward_dim = 32, 
                 num_head = 1,
                 transformer_num_layers = 1,
                 dropout = 0.3,
                 max_len = 1) -> None:
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
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model = hidden_size, 
            dropout = dropout, 
            max_len = max_len
        )
        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, 
            num_head, 
            feedforward_dim, 
            dropout, 
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            transformer_num_layers,
        )
        # 输出层
        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, output_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化影藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        # (seq_len, batch_num, embedding_dim)
        output = self.positional_encoding(output)
        # (seq_len, batch_num, embedding_dim)
        output = self.transformer(output)
        # 将每个词的输出向量取均值，也可以随意取一个标记输出结果，维度为(batch_size, embedding_dim)
        output = output.mean(axis = 1)
        # 输出层 (batch_size, TODO)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        return output





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
