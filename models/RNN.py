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
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model_todo(nn.Module):

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


class Model(nn.Module):

    def __init__(self, args) -> None:
        super(Model, self).__init__()

        self.feature_size = args.feature_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.target_size = args.target_size
        self.pred_len = args.pred_len
        # hideen layer
        self.hidden = nn.Linear(
            in_features = args.feature_size, 
            out_features = args.hidden_size,
        )
        # relu
        self.relu = nn.ReLU()
        # rnn
        self.rnn = nn.RNN(
            input_size = args.hidden_size, 
            hidden_size = args.hidden_size, 
            num_layers = args.num_layers, 
            bias = True,
            batch_first = True,
        )
        # fc layer
        self.linear = nn.Linear(
            in_features = args.hidden_size, 
            out_features = args.target_size,
            bias = True,
        )

    def forward(self, x, hidden = None):
        # [batch_size, obs_len, feature_size]
        batch_size, obs_len, feature_size = x.shape
        # [batch_size, obs_len, hidden_size]
        x_concat = self.hidden(x)
        # [batch_size, obs_len-1, hidden_size]
        H = torch.zeros(batch_size, obs_len-1, self.hidden_size).to(self.args.device)
        # [num_layers, batch_size, hidden_size]
        ht = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.args.device)
        for t in range(obs_len):
            # [batch_size, 1, hidden_size]
            xt = x_concat[:, t, :].viwe(batch_size, 1, -1)
            # ht: [num_layers, batch_size, hidden_size]
            out, ht = self.rnn(xt, ht)
            # [batch_size, hidden_size]
            htt = ht[-1, :, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        # [batch_size, obs_len-1, hidden_size]
        H = self.relu(H)
        x = self.linear(H)

        return x[:, -self.pred_len, :]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
