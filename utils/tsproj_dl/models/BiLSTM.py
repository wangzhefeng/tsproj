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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


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
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # (batch_size, seq_len, 2*hidden_size)
        # 全连接层
        output = self.linear(output)  # (batch_size, seq_len, output_size)
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
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))  # output: (batch_size, seq_len, 2*hidden_size)
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
        )  # output.shape: (batch_size, seq_len, 2*hidden_size)
        # 转换维度
        output = output.reshape(
            output.shape[0], output.shape[1], 2, self.hidden_size
        )  # (batch_size, seq_len, 2, hidden_size)
        # 池化
        output = output.mean(dim = 2)
        # 全连接层
        output = self.linear(output)
        # 只需要返回最后一个时间片的数据
        return output[:, -1, :]




# 测试代码 main 函数
def main():
    from tsproj_dl.config.bilstm import Config
    from data_provider.data_loader_dl import Data_Loader
    from exp.exp_forecasting_dl import train, plot_train_results

    # config
    config = Config()
    
    # data
    data_loader = Data_Loader(cfgs = config)
    train_loader, test_loader = data_loader.run()

    # model
    model = Model_V1(
        feature_size = config.feature_size,
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        output_size = config.output_size,
    )
    
    # loss
    loss_func = nn.MSELoss()
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    
    # model train
    (y_train_pred, y_train_true), (y_test_pred, y_test_true) = train(
        config = config,
        train_loader = train_loader,
        test_loader = test_loader,
        model = model,
        loss_func = loss_func,
        optimizer = optimizer,
        x_train_tensor = data_loader.x_train_tensor, 
        y_train_tensor = data_loader.y_train_tensor,
        x_test_tensor = data_loader.x_test_tensor,
        y_test_tensor = data_loader.y_test_tensor,
        plot_size = 200,
        scaler = data_loader.scaler,
    )
    
    # result plot
    plot_train_results(y_train_pred, y_train_true)
    plot_train_results(y_test_pred, y_test_true)

if __name__ == "__main__":
    main()
