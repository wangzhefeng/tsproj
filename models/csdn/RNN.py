# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RNN.py
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

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )
        self.linear = nn.Linear(in_features = hidden_size, out_features = output_size)

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        # RNN
        output, h_n = self.rnn(x, h_0)
        # 全连接层
        output = self.linear(output)

        output = output[:, -1, :]
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
