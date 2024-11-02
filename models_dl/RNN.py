# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052816
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128619443
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


class Model(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # rnn
        self.rnn = nn.RNN(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True
        )
        # fc layer
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size
        )

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
    from models_dl.config.config_wind_rnn import Config
    from data_provider.data_loader import Data_Loader
    from exp.exp_models_dl import train, plot_train_results

    # config
    config = Config()
    
    # data
    data_loader = Data_Loader(cfg = config)
    train_dataloader, test_dataloader = data_loader.build_dataloader()

    # model
    model = Model(
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
        train_loader = train_dataloader,
        test_loader = test_dataloader,
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
