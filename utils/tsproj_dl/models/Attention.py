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

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


class Model(nn.Module):
    
    def __init__(self, feature_size, seq_len, num_heads, output_size) -> None:
        super(Model, self).__init__()
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim = feature_size,
            num_heads = num_heads,
        )
        # 输出层
        self.linear1 = nn.Linear(
            in_features = feature_size * seq_len, 
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
    from tsproj_dl.config.attn import Config
    from data_provider.data_loader_dl import Data_Loader
    from exp.exp_forecasting_dl import train, plot_train_results

    # config
    config = Config()
    
    # data
    data_loader = Data_Loader(cfgs = config)
    train_loader, test_loader = data_loader.run()

    # model
    model = Model(
        feature_size = config.feature_size,
        seq_len = config.seq_len,
        # hidden_size = config.hidden_size,
        num_heads = config.num_heads,
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
