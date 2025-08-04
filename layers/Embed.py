# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Embed.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-04
# * Version     : 0.1.110414
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import numpy as np
import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TokenEmbedding(nn.Module):
    """
    Token Embedding
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        
        # Conv1d layer
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels = c_in,
            out_channels = d_model,
            kernel_size = 3,
            padding = padding,
            padding_mode = 'circular',
            bias = False,
        )
        # Conv1d layer 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)

        return x


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding
    """

    def __init__(self, d_model, max_len = 5000):
        super(PositionalEmbedding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]

        return x


class PositionalEmbedding_v2(nn.Module):
    
    def __init__(self, d_model, n_position=1024):
        super(PositionalEmbedding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):

        return self.pos_table[:, :x.size(1)].clone().detach()


class FixedEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad = False)

    def forward(self, x):
        x = self.emb(x).detach()

        return x


class TemporalEmbedding(nn.Module):
    """
    Temporal Embedding
    """

    def __init__(self, d_model, embed_type = 'fixed', freq = 'h'):
        super(TemporalEmbedding, self).__init__()
        
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Time Feature Embedding

    args.freq: freq for time features encoding, 
    options: [
        s:secondly, t:minutely, h:hourly, d:daily, 
        b:business days, w:weekly, m:monthly
    ], 
    you can also use more detailed freq like 15min or 3h'
    """

    def __init__(self, d_model, embed_type = 'timeF', freq = 'h'):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {
        #     'h': 4, 
        #     't': 5, 
        #     's': 6,
        #     'm': 1, 
        #     'a': 1,
        #     'w': 2, 
        #     'd': 3, 
        #     'b': 3,
        # }
        # d_inp = freq_map[freq]
        
        def freq_to_dim(freq):
            """
            https://github.com/thuml/Time-Series-Library/pull/261
            """
            while freq[0].isdigit():
                freq = freq[1:]
            freq = freq.lower()
            if freq == "min":
                freq = 't'
            elif freq == "A":
                freq = 'y'

            freq_map = {
                'h': 4,  # hourly
                't': 5,  # minutely
                's': 6,  # secondly
                'm': 1,  # monthly
                'a': 1,  # TODO
                'w': 2,  # weekly
                'd': 3,  # daily
                'b': 3,  # business days
            }
            return freq_map[freq]
        d_inp = freq_to_dim(freq)
        
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        x = self.embed(x)

        return x


class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding, self).__init__()

        # value embedding
        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        # position embedding
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        # temporal embedding
        if embed_type == "timeF":
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model = d_model, 
                embed_type = embed_type, 
                freq = freq,
            )
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model = d_model, 
                embed_type = embed_type, 
                freq = freq,
            )
        # dropout
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        # embedding
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # dropout
        x = self.dropout(x)

        return x


class DataEmbedding_inverted(nn.Module):

    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding_inverted, self).__init__()

        # value embedding
        self.value_embedding = nn.Linear(c_in, d_model)
        # dropout
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        x = self.dropout(x)
        
        return x


class DataEmbedding_wo_pos(nn.Module):
    """
    Data Embedding wo pos
    """

    def __init__(self, c_in, d_model, embed_type = 'fixed', freq = 'h', dropout = 0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in = c_in, d_model = d_model)
        self.position_embedding = PositionalEmbedding(d_model = d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model = d_model, 
            embed_type = embed_type,
            freq = freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model = d_model, 
            embed_type = embed_type, 
            freq = freq
        )
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)        
        x = self.dropout(x)

        return x


class PatchEmbedding(nn.Module):
    
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias = False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars




# 测试代码 main 函数
def main():
    x = torch.randn(1, 50, 5)
    x = x.permute(0, 2, 1)
    print(x)
    print(x.size())
    x = x.transpose(1, 2)
    print(x.size())

    # ------------------------------
    # conv1d
    # ------------------------------
    conv1 = nn.Conv1d(in_channels=50, out_channels=5, kernel_size=3) 
    conv1_out = conv1(x)
    # conv1_out = conv1_out.transpose(1, 2)
    print(conv1_out)
    print(conv1_out.size())

    # ------------------------------
    # token embedding
    # ------------------------------
    # token_embed = TokenEmbedding(c_in = 7, d_model = 512)
    # print(token_embed)

    # res = token_embed(x)
    # print(res)

if __name__ == "__main__":
    main()
