# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052722
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
from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# config 单变量
class Config_Univariate_SingleOutput:
    data_path = "data/wind_dataset.csv"
    timestep = 1  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "LSTM"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# config 多变量
class Config_MultiVariate_SingleOutput:
    data_path = "data/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "TODO"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


class Config_MultiVariate_MultiOutput:
    data_path = "data/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 2  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 1e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "TODO"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


config = Config_Univariate_SingleOutput()


class Model(nn.Module):
    
    def __init__(self, feature_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(Model, self).__init__()
        self.hidden_size = hidden_size  # 影藏层大小
        self.num_layers = num_layers  # LSTM 层数
        self.lstm = nn.LSTM(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        # 获取批次大小
        batch_size = x.shape[0]  # x = [batch_size, ]
        # 初始化隐藏状态
        # h_0 = torch.randn(self.num_layers, len(x), self.hidden_size)
        # c_0 = torch.randn(self.num_layers, len(x), self.hidden_size)
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # LSTM
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        # 全连接层
        output = self.linear(output)  # (batch_size * timestep, 1)
        # 返回最后一个时间片的数据
        output = output[: -1, :]
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
