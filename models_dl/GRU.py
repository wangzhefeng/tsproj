# -*- coding: utf-8 -*-

# ***************************************************
# * File        : GRURegressor.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-25
# * Version     : 0.1.052521
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

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = hidden_size  # 影藏层大小
        self.num_layers = num_layers  # GRU 层数
        # GRU
        self.gru = nn.GRU(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )
        # fc
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size
        )

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化影藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        # GRU
        output, h_0 = self.gru(x, h_0)
        # 获取 GRU 输出的维度信息
        batch_size, timestep, hidden_size = output.shape
        # 将 output 变成 (batch_size * timestep, hidden_dim)
        output = output.reshape(-1, hidden_size)
        # 全连接层
        output = self.linear(output)  # shape: (batch_size * timestep, 1)
        # 转换维度，用于输出
        output = output.reshape(timestep, batch_size, -1)
        # 只需返回最后一个时间片的数据
        return output[-1]



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
