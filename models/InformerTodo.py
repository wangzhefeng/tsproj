# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Informer.py
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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_encoder_layers, 
                 num_decoder_layers,
                 d_model,
                 nhead, 
                 dim_feedforward,
                 dropout,
                 activation = "gelu") -> None:
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropput = dropout
        self.activation = activation

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = num_encoder_layers,
        )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = num_decoder_layers,
        )
        # full connect
        self.linear = nn.Linear(in_features = d_model, out_features = output_size)
    
    def forward(self, x):
        # 将数据维度从 (batch_size, seq_len, input_size) 变为 (seq_len, batch_size, input_size)
        x = x.permute(1, 0, 2)
        # encoder
        enc_output = self.encoder(x)
        # decoder: 解码器将编码器输出作为输入，并且预测目标序列
        dec_output = self.decoder(enc_output, enc_output)
        # 取最后一个时间步的输出，并通过全连接层得到最终输出
        output = self.linear(dec_output[-1])
        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
