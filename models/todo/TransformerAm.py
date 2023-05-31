# -*- coding: utf-8 -*-

# ***************************************************
# * File        : simple_transformer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-30
# * Version     : 0.1.033021
# * Description : 简单 Transformer 模型用于时序数据的预测
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import math

import torch
import torch.nn as nn
torch.manual_seed(0)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    position encoding for transformer
    """
    def __init__(self, d_model, max_len = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # max_len, 1, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerAm(nn.Module):
    
    def __init__(self, feature_size = 250, num_head = 10, num_layers = 1, dropout = 0.1):
        """
        包含 encoder 和 decoder 的 transformer 模型

        Parameters:
            feature_size: d_model, int
            num_head: 注意力机制头数, int
            num_layers: encoder layer 层数
            dropout: float
        """
        super(TransformerAm, self).__init__()
        # self.model_type = 'TransformerForecasting'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)  # d_model = feature_size
        self.encoder = nn.TransformerEncoderLayer(
            d_model = feature_size, 
            nhead = num_head, 
            dropout = dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers = num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        """
        模型参数初始化
        """
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)  # input_window, batch_size, fea_size
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 返回矩阵上三角部分
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
