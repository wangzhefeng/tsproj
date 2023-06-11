# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Attention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052817
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
    
    def __init__(self, feature_size, timestep, num_heads, output_size) -> None:
        super(Model, self).__init__()
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim = feature_size,
            num_heads = num_heads,
        )
        # 输出层
        self.linear1 = nn.Linear(
            in_features = feature_size * timestep, 
            out_features = 256
        )
        self.linear2 = nn.Linear(
            in_features = 256,
            out_features = output_size,
        )
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, query, key, value):
        # 注意力
        attn_output, attn_output_weights = self.attention(query, key, value)   
        # 展开
        output = attn_output.flatten(start_dim = 1)  # (32, 20)
        # 全连接层
        output = self.linear1(output)  # (32, 256)
        output = self.relu(output)
        output = self.linear2(output)  # (32, output_size)
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
