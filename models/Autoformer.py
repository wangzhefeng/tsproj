# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Autoformer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041919
# * Description : Autoformer is the first method to achieve the series-wise connection,
# *               with inherent O(LlogL) complexity
# * Link        : Paper link: https://openreview.net/pdf?id=I55UqU-M11y
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Decoder, DecoderLayer, 
    Encoder, EncoderLayer, 
    my_Layernorm,
    series_decomp
)
from layers.Embed import DataEmbedding_wo_pos

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # params
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout = configs.dropout, output_attention = configs.output_attention),
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
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in, 
                configs.d_model, 
                configs.embed, 
                configs.freq, 
                configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout = configs.dropout, output_attention = False),
                            configs.d_model, 
                            configs.n_heads
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout = configs.dropout, output_attention = False),
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
                projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
            )
        
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
        
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
        
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init 
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        mean = torch.mean(x_enc, dim = 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device = x_enc.device)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim = 1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim = 1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
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
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
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
