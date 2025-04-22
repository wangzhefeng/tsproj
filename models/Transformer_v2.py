# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Transformer_v2.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-22
# * Version     : 1.0.042210
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.TransformerBlocks import Encoder, Decoder
from layers.Invertible import RevIN


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(
                configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in, configs.d_model, configs.dropout
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in, configs.d_model, configs.dropout
            )
        # Encoder
        self.encoder = Encoder(
            configs.e_layers, 
            configs.n_heads, 
            configs.d_model, 
            configs.d_ff, 
            configs.dropout, 
            configs.activation, 
            configs.output_attention,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            configs.d_layers, 
            configs.n_heads, 
            configs.d_model, 
            configs.d_ff,
            configs.dropout, 
            configs.activation, 
            configs.output_attention,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # forward-feed layer
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        # normalize and denormalize
        self.rev = RevIN(configs.c_out) if configs.rev else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # _normalize
        x_enc = self.rev(x_enc, 'norm') if self.rev else x_enc
        
        # Embedding
        enc_embed = self.enc_embedding(x_enc, x_mark_enc)
        # Encoder
        enc_out, attns = self.encoder(enc_embed, attn_mask=enc_self_mask)
        
        # Embedding
        dec_embed = self.dec_embedding(x_dec, x_mark_dec)
        # Decoder
        dec_out = self.decoder(dec_embed, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        # forward-feed layer
        proj = self.projection(dec_out)
        # denormalize
        output = self.rev(proj, 'denorm') if self.rev else proj
        # output
        if self.output_attention:
            return output[:, -self.pred_len:, :], attns
        else:
            return output[:, -self.pred_len:, :]  # [B, L, D] <=> [batch_size, pred_len, c_out]
