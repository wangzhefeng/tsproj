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

from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    import torch
    from data_provider.data_splitor import Datasplitor
    from experiments.exp_csdn import model_train
    from utils.plot_results import plot_train_results
    from config.config_wind_lstm import Config_Univariate_SingleOutput_V1

    # ------------------------------
    # config
    # ------------------------------
    config = Config_Univariate_SingleOutput_V1()
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
