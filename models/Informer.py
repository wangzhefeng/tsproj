# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Informer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-05
# * Version     : 0.1.050512
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

from layers.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.decoder import Decoder, DecoderLayer
from layers.attn import FullAttention, ProbAttention, AttentionLayer
from layers.embedding import DataEmbedding

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Informer(nn.Module):

    def __init__(self, 
                 enc_in, 
                 dec_in, 
                 c_out, 
                 seq_len, label_len, 
                 out_len, 
                 factor = 5, 
                 d_model = 512, 
                 n_heads = 8, 
                 e_layers = 3, 
                 d_layers = 2, 
                 d_ff = 512, 
                 dropout = 0.0, 
                 attn = 'prob', 
                 embed = 'fixed', 
                 freq = 'h', 
                 activation = 'gelu', 
                 output_attention = False, 
                 distil = True, 
                 mix = True,
                 device = torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout = dropout, output_attention = output_attention), 
                        d_model, 
                        n_heads, 
                        mix = False,
                    ),
                    d_model,
                    d_ff,
                    dropout = dropout,
                    activation = activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer = nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, 
                        n_heads, 
                        mix = mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout = dropout, output_attention = False), 
                        d_model, 
                        n_heads, 
                        mix = False
                    ),
                    d_model,
                    d_ff,
                    dropout = dropout,
                    activation = activation,
                )
                for l in range(d_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias = True)
     
    def forward(self, 
                x_enc, 
                x_mark_enc, 
                x_dec, 
                x_mark_dec, 
                enc_self_mask = None, 
                dec_self_mask = None, 
                dec_enc_mask = None):
        # TODO
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask = enc_self_mask)
        # TODO
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask = dec_self_mask, cross_mask = dec_enc_mask)
        dec_out = self.projection(dec_out)
        # TODO 
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):

    def __init__(self, 
                 enc_in, 
                 dec_in, 
                 c_out, 
                 seq_len, 
                 label_len, 
                 out_len, 
                 factor = 5, 
                 d_model = 512, 
                 n_heads = 8, 
                 e_layers = [3, 2, 1], 
                 d_layers = 2, 
                 d_ff = 512, 
                 dropout = 0.0, 
                 attn = 'prob', 
                 embed = 'fixed', 
                 freq = 'h', 
                 activation = 'gelu',
                 output_attention = False, 
                 distil = True, 
                 mix = True,
                 device = torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                            d_model, 
                            n_heads, 
                            mix = False
                        ),
                        d_model,
                        d_ff,
                        dropout = dropout,
                        activation = activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer = nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout = dropout, output_attention = False), 
                        d_model, 
                        n_heads, 
                        mix = mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout = dropout, output_attention = False), 
                        d_model, 
                        n_heads, 
                        mix = False
                    ),
                    d_model,
                    d_ff,
                    dropout = dropout,
                    activation = activation,
                )
                for l in range(d_layers)
            ],
            norm_layer = nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias = True)
        
    def forward(self, 
                x_enc, x_mark_enc, 
                x_dec, x_mark_dec, 
                enc_self_mask = None, 
                dec_self_mask = None, 
                dec_enc_mask = None):
        # TODO
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask = enc_self_mask)
        # TODO
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask = dec_self_mask, cross_mask = dec_enc_mask)
        dec_out = self.projection(dec_out)
        # TODO
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
