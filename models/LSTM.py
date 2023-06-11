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


# 单变量-单输出
class Config_Univariate_SingleOutput_V1:
    data_path = "dataset/wind_dataset.csv"
    timestep = 1  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # LSTM 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "LSTM-Univariate-SingleOutput-V1"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# TODO
class Config_Univariate_SingleOutput_V2:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "LSTM"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# 多变量-单步输出
class Config_MultiVariate_SingleOutput:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = None  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "Config_MultiVariate_SingleOutput"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# 多变量-多步输出
class Config_MultiVariate_MultiOutput:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 2  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 1e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "TODO"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


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
        batch_size = x.shape[0]  # x.shape=(batch_size, timestep, feature_size)

        # 初始化隐藏状态
        if hidden is None:
            h_0 = x.data.new(
                self.num_layers, batch_size, self.hidden_size
            ).fill_(0).float()  # (D*num_layers, batch_size, hidden_size)
            c_0 = x.data.new(
                self.num_layers, batch_size, self.hidden_size
            ).fill_(0).float()  # (D*num_layers, batch_size, hidden_size)
        else:
            h_0, c_0 = hidden

        # LSTM
        # output.shape=(batch_size, timestep, D*output_size) 
        # h_0.shape=(D*num_layers, batch_size, h_out)
        # c_0.shape=(D*num_layers, batch_size, h_cell)
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        batch_size, timestep, hidden_size = output.shape  # 获取 LSTM 输出的维度信息
        output = output.reshape(-1, hidden_size)  # 将 output 变成 (batch_size * timestep, hidden_size)

        # 全连接层
        output = self.linear(output)  # (batch_size * timestep, 1)
        output = output.reshape(timestep, batch_size, -1)  # 转换维度用于输出

        # 返回最后一个时间片的数据
        # output = output[: -1, :]
        output = output[-1]
        return output




# 测试代码 main 函数
def main():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from data_provider.data_splitor import Datasplitor
    from experiments.exp_csdn import model_train
    from utils.plot_results import plot_train_results

    # ------------------------------
    # config
    # ------------------------------
    config = Config_MultiVariate_SingleOutput()
    # ------------------------------
    # data
    # ------------------------------
    df = pd.read_csv(config.data_path, index_col = 0)
    print(df.head())
    # ------------------------------
    # data preprocess
    # ------------------------------
    scaler = MinMaxScaler()
    scaler.fit_transform(np.array(df["WIND"]).reshape(-1, 1))
    scaler_model = MinMaxScaler()
    data = scaler_model.fit_transform(np.array(df))
    # ------------------------------
    # data split
    # ------------------------------ 
    data_split = Datasplitor(
        data = data,
        timestep = config.timestep,
        feature_size = config.feature_size, 
        output_size = config.output_size, 
        target_index = config.target_index,
        split_ratio = config.split_ratio,
        batch_size = config.batch_size,
    )
    train_data, train_loader, \
    test_data, test_loader = data_split.RecursiveMultiStep()
    # ------------------------------
    # model
    # ------------------------------
    # model = Model(
    #     feature_size = config.feature_size,
    #     hidden_size = config.hidden_size,
    #     num_layers = config.num_layers,
    #     output_size = config.output_size,
    # )
    # loss_func = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    # y_train_pred, y_test_pred = model_train(
    #     config = config,
    #     train_loader = train_loader, 
    #     test_loader = test_loader, 
    #     model = model, 
    #     loss_func = loss_func, 
    #     optimizer = optimizer
    # )

    # train_pred = scaler.inverse_transform(model(y_test_pred).detach().numpy()[:plot_size]).reshape(-1, 1)
    # train_true = scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size])
    # test_pred = scaler.inverse_transform(y_test_pred.detach().numpy()[:plot_size])
    # test_true = scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size])
    # plot_train_results(pred = train_pred, true = train_true)
    # plot_test_results(pred = test_pred, true = test_true)

if __name__ == "__main__":
    main()
