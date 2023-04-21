# -*- coding: utf-8 -*-


# ***************************************************
# * File        : transform_forecast.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-30
# * Version     : 0.1.033021
# * Description : pytorch 标准 transform 用于时序数据预测
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import logging
import math
import time
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from TransformerAm import TransformerAm


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformForecasting:
    
    def __init__(self, 
                 d_model,
                 num_head,
                 num_layers,
                 input_wsz = 100,
                 output_wsz = 5,
                 train_size = 0.8,
                 batch_size = 20,
                 dropout = 0.1,
                 lr = 0.005):
        # params
        self.input_wsz = input_wsz  # number of input steps
        self.output_wsz = output_wsz  # number of prediction steps
        self.train_size = train_size  # train data factor
        self.batch_size = batch_size  # batch size
        # data transformer
        self.scaler = MinMaxScaler(feature_range = (-1, 1))
        # model
        self.model = TransformerAm(
            feature_size = d_model,
            num_head = num_head,
            num_layers = num_layers,
            dropout = dropout
        ).to(device)  # feature_size = 250, num_head = 10, num_layers = 1, dropout = 0.1
        # loss
        self.loss_fn = torch.nn.MSELoss()
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)
        # lr scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma = 0.95)

    def Train(self, series_data, epochs = 100):
        train_data, val_data = self.GetData(series_data)
        for epoch_i in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.EpochTrain(train_data, epoch_i)
            val_loss = self.Validate(val_data)
            logging.info("-" * 89)
            epoch_info = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch_i, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
            logging.info(epoch_info)
            logging.info("-" * 89)
            self.scheduler.step()

    def EpochTrain(self, train_data, epoch_i):
        self.model.train()
        total_loss = 0.
        log_interval = int(len(train_data) / self.batch_size / 5)

        start_time = time.time()

        for batch, i in enumerate(range(0, len(train_data) - 1, self.batch_size)):
            data, target = self.GetBatch(train_data, i, self.batch_size)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                log_info = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'. \
                    format(epoch_i, batch, len(train_data) // self.batch_size, self.scheduler.get_lr()[0],
                           elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss))
                logging.info(log_info)
                total_loss = 0
                start_time = time.time()

    def Validate(self, validate_data):
        self.model.eval()
        total_loss = 0.
        eval_batch_size = 100
        with torch.no_grad():
            for i in range(0, len(validate_data) - 1, eval_batch_size):
                data, targets = self.GetBatch(validate_data, i, eval_batch_size)
                output = self.model(data)
                total_loss += len(data[0]) * self.criterion(output, targets).cpu().item()
        return total_loss / len(validate_data)

    def Predict(self, data_source, steps = 10):
        self.model.eval()
        data, _ = self.GetBatch(data_source, 0, 1)
        with torch.no_grad():
            for i in range(0, steps):
                output = self.model(data[-self.input_wsz:])
                data = torch.cat((data, output[-1:]))
        return data

    def GetBatch(self, source_data, i, batch_size):
        seq_len = min(batch_size, len(source_data) - 1 - i)
        data = source_data[i:i + seq_len]
        input_data = torch.stack(torch.stack([item[0] for item in data]).chunk(self.input_wsz, 1))
        target_data = torch.stack(torch.stack([item[1] for item in data]).chunk(self.input_wsz, 1))
        return input_data, target_data

    def GetData(self, series: pd.Series):
        """
        将时序数据转为训练样本
        """
        # data scale
        amplitude = self.scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
        # data split
        train_len = int(len(series) * self.train_size)
        train_data = amplitude[:train_len]
        test_data = amplitude[train_len:]
        # data sequence
        train_sequence = self.CreateIOSequences(train_data)
        train_sequence = train_sequence[:-self.output_wsz].to(device)
        test_sequence = self.CreateIOSequences(test_data)
        test_sequence = test_data[:-self.output_wsz].to(device)

        return train_sequence, test_sequence

    def CreateIOSequences(self, series_data: Union[List, pd.Series]):
        """
        convert series data to train sequences
        """
        io_seq = []
        L = len(series_data)
        tw = self.input_wsz
        for i in range(L - tw):
            train_seq = np.append(series_data[i:i + tw][:-self.output_wsz], self.output_wsz * [0])
            train_label = series_data[i:i + tw]
            io_seq.append((train_seq, train_label))
        return torch.FloatTensor(io_seq)

    def Load(self, param_path, model_path):
        data = np.load(param_path)
        self.input_wsz = data['input_wsz']
        self.output_wsz = data['output_wsz']
        self.train_size = data['train_size']
        self.batch_size = data['batch_size']

        self.model = torch.load(model_path)

    def Save(self, param_path, model_path):
        np.savez(
            param_path,
            input_wsz = self.input_wsz,
            output_wsz = self.output_wsz,
            train_size = self.train_size,
            batch_size = self.batch_size
        )
        torch.save(self.model, model_path)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
