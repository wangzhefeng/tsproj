# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052722
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128605806
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


class LSTM(nn.Module):
    
    def __init__(self, feature_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # lstm
        self.lstm = nn.LSTM(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )
        # fc layer
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        # 获取批次大小
        batch_size = x.shape[0]  # x.shape=(batch_size, seq_len, feature_size)
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
        # output.shape=(batch_size, seq_len, D*output_size) 
        # h_0.shape=(D*num_layers, batch_size, h_out)
        # c_0.shape=(D*num_layers, batch_size, h_cell)
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        batch_size, seq_len, hidden_size = output.shape  # 获取 LSTM 输出的维度信息
        output = output.reshape(-1, hidden_size)  # 将 output 变成 (batch_size * seq_len, hidden_size)

        # 全连接层
        output = self.linear(output)  # (batch_size * seq_len, 1)
        # TODO output = output.reshape(seq_len, batch_size, -1)  # 转换维度用于输出
        output = output.reshape(batch_size, seq_len, -1).permute(1, 0, 2)  # 转换维度用于输出

        # 返回最后一个时间片的数据
        # output = output[: -1, :]
        output = output[-1]

        return output




# 测试代码 main 函数
def main():
    from tsproj_dl.config.lstm import Config
    from tsproj_dl.data_provider.data_loader import Data_Loader
    from tsproj_dl.exp.exp_forecasting import train, plot_train_results
    # config
    config = config()
    
    # data
    data_loader = Data_Loader(cfgs = config)
    train_loader, test_loader = data_loader.run()

    # model
    model = LSTM(
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
