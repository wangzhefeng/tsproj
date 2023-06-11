# -*- coding: utf-8 -*-

# ***************************************************
# * File        : BiLSTM.py
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


class Model_V1(nn.Module):
    
    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model_V1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = feature_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True, 
            bidirectional = True,
        )
        self.linear = nn.Linear(
            in_features = 2 * hidden_size,
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏状态
        if hidden is None:
            h_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # (batch_size, timestep, 2*hidden_size)
        # 全连接层
        output = self.linear(output)  # (batch_size, timestep, output_size)
        return output[: -1, :]


class Model_V2(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model_V2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = feature_size,
            hidden_size = hidden_size,
            batch_first = True,
            bidirectional = True,
        )
        self.linear = nn.Linear(
            in_features = 2 * hidden_size,
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # output: (batch_size, timestep, 2*hidden_size)
        output = torch.cat([
            output[:, -1, :self.hidden_size], 
            output[:, 0, self.hidden_size:]
        ], dim = 1)  # (batch_size, 2*hidden_size)
        # 全连接层
        output = self.linear(output)  # (batch_size, output_size)
        return output

    
class Model_V3(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model_V3, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = feature_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True,
        )
        self.linear = nn.Linear(
            in_features = hidden_size,
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(2 * self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(
            x, (h_0, c_0)
        )  # output.shape: (batch_size, timestep, 2*hidden_size)
        # 转换维度
        output = output.reshape(
            output.shape[0], output.shape[1], 2, self.hidden_size
        )  # (batch_size, timestep, 2, hidden_size)
        # 池化
        output = output.mean(dim = 2)
        # 全连接层
        output = self.linear(output)
        # 只需要返回最后一个时间片的数据
        return output[:, -1, :]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
