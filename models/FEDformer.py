# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FEDformer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-04
# * Version     : 0.1.110400
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Decoder, DecoderLayer, 
    Encoder, EncoderLayer, 
    my_Layernorm,
    series_decomp
)
from layers.Embed import DataEmbedding
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version = 'fourier', mode_select = 'random', modes = 32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        # params
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        # Time Series Decomp
        self.decomp = series_decomp(configs.moving_avg)
        # Data Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, 
            configs.d_model,
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        # 小波变换或傅里叶变换
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich = configs.d_model, L = 1, base = 'legendre')
            decoder_self_att = MultiWaveletTransform(ich = configs.d_model, L = 1, base = 'legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels = configs.d_model,
                out_channels = configs.d_model,
                seq_len_q = self.seq_len // 2 + self.pred_len,
                seq_len_kv = self.seq_len,
                modes = self.modes,
                ich = configs.d_model,
                base = 'legendre',
                activation = 'tanh'
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels = configs.d_model,
                out_channels = configs.d_model,
                seq_len = self.seq_len,
                modes = self.modes,
                mode_select_method = self.mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels = configs.d_model,
                out_channels = configs.d_model,
                seq_len = self.seq_len // 2 + self.pred_len,
                modes = self.modes,
                mode_select_method = self.mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels = configs.d_model,
                out_channels = configs.d_model,
                seq_len_q = self.seq_len // 2 + self.pred_len,
                seq_len_kv = self.seq_len,
                modes = self.modes,
                mode_select_method = self.mode_select,
                num_heads = configs.n_heads
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout = configs.dropout,
                    activation = configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer = my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, 
                        configs.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout = configs.dropout,
                    activation = configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer = my_Layernorm(configs.d_model),
            projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        # Other tasks
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init 
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        mean = torch.mean(x_enc, dim = 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # data embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # enc
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # dec
        seasonal_part, trend_part = self.decoder(
            dec_out, 
            enc_out, 
            x_mask = None, 
            cross_mask = None, 
            trend = trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
