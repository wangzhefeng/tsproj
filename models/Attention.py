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


class Config:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_heads = 1  # 注意力机制头的数量
    # num_layers = 2  # 网络的层数
    # hidden_size = 256  # 网络隐藏层大小
    output_size = 1  # 预测未来 n 个时刻数据
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "Attention"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"

config = Config()


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
