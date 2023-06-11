# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Seq2Seq_LSTM.py
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


class Encoder(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )

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
        return output, h_0, c_0


class Decoder(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # lstm
        self.lstm = nn.LSTM(
            input_size = feature_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )
        # linear
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size
        )

    def forward(self, h_0, c_0):
        x = torch.zeros(h_0.shape[1], self.output_size, self.hidden_size)
        # LSTM
        output, _ = self.lstm(x, (h_0, c_0))
        # Linear
        output = self.linear(output)
        # 输出层
        output = output[:, -1, :]
        return output


class Seq2Seq(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(feature_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, output_size)

    def forward(self, x):
        _, h_n, c_n = self.encoder(x)
        output = self.decoder(h_n, c_n)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
