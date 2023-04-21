# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LSTM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032806
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LSTMRegressor(nn.Module):

    def __init__(self, num_features, hidden_dim = 256, num_layers = 2) -> None:
        super(LSTMRegressor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size = num_features,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
        )
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        hidden = torch.randn(self.num_layers, len(x), self.hidden_dim)
        carry = torch.randn(self.num_layers, len(x), self.hidden_dim)
        output, (hidden, carry) = self.lstm(x, (hidden, carry))
        out = self.linear(output[:, -1])
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
