# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MLP.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052816
# * Description : description
# * Link        : https://mp.weixin.qq.com/s?__biz=MzkzMTMyMDQ0Mw==&mid=2247485132&idx=2&sn=70547a02346b21bd1682733d6fb8ee2e&chksm=c26d81d8f51a08ced66dcbd612e3c77ea75c7308a5457dc5d7bf3ea45fa551bdb868316f7344&scene=132#wechat_redirect
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, feature_size, seq_len, hidden_size, output_size):
        super(Model, self).__init__()
        # 全连接层
        self.linear1 = nn.Linear(in_features = feature_size * seq_len, out_features = hidden_size[0])
        self.linear2 = nn.Linear(in_features = hidden_size[0], out_features = hidden_size[1])
        self.linear3 = nn.Linear(in_features = hidden_size[1], out_features = hidden_size[2])
        self.linear4 = nn.Linear(in_features = hidden_size[2], out_features = output_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
    
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.relu(x)

        output = self.linear4(x)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
