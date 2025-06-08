# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM2LSTM.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-08
# * Version     : 1.0.060819
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class LSTMEncoder(nn.Module):

    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        
        self.num_layers = rnn_num_layers
        self.input_feature_len = input_feature_len
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.rnn_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, input_seq):
        ht = torch.zeros(
            self.num_layers * self.rnn_directions, 
            input_seq.size(0), 
            self.hidden_size, 
            device=input_seq.device,
        )
        ct = ht.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out, (ht, ct) = self.lstm(input_seq, (ht,ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            lstm_out = torch.sum(lstm_out, axis=2)
        
        return lstm_out, ht.squeeze(0)


class AttentionDecoderCell(nn.Module):
    
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(
            hidden_size + input_feature_len, 
            sequence_len
        )
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(
            hidden_size, 
            input_feature_len
        )

    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden, rnn_hidden = self.decoder_rnn_cell(attention_combine, (prev_hidden, prev_hidden))
        output = self.out(rnn_hidden)
        
        return output, rnn_hidden


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.input_size = args.feature_size
        self.output_size = args.target_size
        self.pred_len = args.pred_len
        self.teacher_forcing = args.teacher_forcing
        self.encoder = LSTMEncoder(args.num_layers, args.feature_size, args.seq_len, args.hidden_size)
        self.decoder_cell = AttentionDecoderCell(args.feature_size, args.target_size, args.seq_len, args.hidden_size)
        self.linear = nn.Linear(args.feature_size, args.target_size)

    def __call__(self, x_batch, y_batch=None):
        # encoder
        encoder_output, encoder_hidden = self.encoder(x_batch)
        # decoder
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(
                self.pred_len, 
                x_batch.size(0), 
                self.input_size, 
                device=x_batch.device
            )
        else:
            outputs = torch.zeros(x_batch.size(0), self.output_size)
        
        y_prev = x_batch[:, -1, :]
        for i in range(self.pred_len):
            if (y_batch is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = y_batch[:, i].unsqueeze(1)
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output
        # linear output
        outputs = outputs.permute(1, 0, 2)
        if self.output_size == 1:
            outputs = self.linear(outputs)
        
        return outputs




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
