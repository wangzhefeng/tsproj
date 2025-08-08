# -*- coding: utf-8 -*-

# ***************************************************
# * File        : GRURegressor.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-25
# * Version     : 0.1.052521
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128595011
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


class Model_v1(nn.Module):
    """
    GRU
    """

    def __init__(self, cfgs) -> None:
        """
        feature_size (_type_): 每个时间点的特征维度
        hidden_size (_type_): GRU 内部隐层的维度
        num_layers (_type_): GRU 层数，默认为 1
        output_size (_type_): 输出维度
        """
        super(Model_v1, self).__init__()
        
        self.cfgs = cfgs
        self.hidden_size = self.cfgs.hidden_size
        self.num_layers = self.cfgs.num_layers
        # GRU
        self.gru = nn.GRU(
            input_size = self.cfgs.feature_size, 
            hidden_size = self.cfgs.hidden_size, 
            num_layers = self.cfgs.num_layers,
            bias = True, 
            batch_first = True,  # [B, seq_len, ]
            dropout = 0,  # 是否采用 dropout
            bidirectional = False,  # 是否采用双向 GRU 模型
            device=self.cfgs.device,
            dtype=self.cfgs.dtype
        )
        self.dropout = nn.Dropout(0.1)
        # fc
        self.linear = nn.Linear(
            in_features = self.cfgs.hidden_size, 
            out_features = self.cfgs.output_size,
            device=self.cfgs.device,
            dtype=self.cfgs.dtype,
        )
        self.relu = nn.ReLU()

    def forward(self, x, hidden = None):
        # 获取 batch size
        batch_size = x.shape[0]

        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        
        # GRU
        output, h_0 = self.gru(x, h_0)

        # TODO dropout
        # output = self.dropout(output)

        # 获取 GRU 输出的维度信息
        batch_size, seq_len, hidden_size = output.shape

        # 将 output 变成 (batch_size * seq_len, hidden_dim)
        output = output.reshape(-1, hidden_size)
        # TODO output = output[:, -self.pred_len:, :]

        # 全连接层
        output = self.linear(output)  # shape: (batch_size * seq_len, 1)
        # TODO output = self.relu(output)

        # 转换维度，用于输出
        # TODO output = output.reshape(seq_len, batch_size, -1)
        output = output.reshape(batch_size, seq_len, -1).permute(1, 0, 2)

        # 只需返回最后一个时间片的数据
        return output[-1]


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.args = args
        self.gru = nn.GRU(
            args.feature_size, 
            args.hidden_size, 
            args.num_layers, 
            bias=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(args.hidden_size, args.output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size, seq_len, feature_size = x.shape

        h0_gru = torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size).to(x.device)
        
        # gru layer
        out, _ = self.gru(x, h0_gru)
        out = self.dropout(out)
        
        # 取最后 pred_len 时间步的输出
        out = out[:, -self.args.pred_len:, :]
        
        # fc linear
        out = self.fc(out)
        # relu
        out = self.relu(out)
        
        return out





# 测试代码 main 函数
def main():
    import torch
    from utils.log_util import logger

    # model
    model = nn.GRU(
        input_size=3, 
        hidden_size=10, 
        num_layers=2, 
        bias=True, 
        batch_first=True, 
        bidirectional=False
    )
    logger.info(model)

    # data
    x = torch.randn(1, 5, 3)

    # forward
    output, h_0 = model(x)
    print(output.shape)
    print(h_0.shape)

if __name__ == "__main__":
    main()
