# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CNN1D.py
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

    def __init__(self, feature_size, out_channels, output_size) -> None:
        super(Model, self).__init__()
        # 一维卷积
        self.conv1d_1 = nn.Conv1d(in_channels = feature_size, out_channels = out_channels[0], kernel_size = 3)
        self.conv1d_2 = nn.Conv1d(in_channels = out_channels[0], out_channels = out_channels[1], kernel_size = 3)
        self.conv1d_3 = nn.Conv1d(in_channels = out_channels[1], out_channels = out_channels[2], kernel_size = 3)
        # 输出层
        self.linear1 = nn.Linear(in_features = out_channels[2] * 14, out_features = 128)
        self.linear2 = nn.Linear(in_features = 128, out_features = output_size)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, feature_size, seq_len[32, 1, 20])

        x = self.conv1d_1(x)  # [32, 10, 18]
        x = self.relu(x)

        x = self.conv1d_2(x)  # [32, 20, 16]
        x = self.relu(x)

        x = self.conv1d_3(x)  # [32, 30, 14]
        x = self.relu(x)

        x = x.flatten(start_dim = 1)  # [32, 420]

        x = self.linear1(x)
        x = self.relu(x)

        output = self.linear2(x)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
