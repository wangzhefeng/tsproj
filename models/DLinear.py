# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DLinear.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-03
# * Version     : 0.1.110323
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

from layers.SeriesDecomp import series_decomp

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual: bool = False):
        super(Model, self).__init__()
        
        # 任务类型
        self.task_name = configs.task_name
        # 输入历史数据长度
        self.seq_len = configs.seq_len
        # 输出未来数据长度
        if self.task_name == 'classification' or \
           self.task_name == 'anomaly_detection' or \
           self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # whether shared model among different variates
        self.individual = individual
        # encoder input size
        self.channels = configs.enc_in
        # 时间序列分解实例：Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        # ------------------------------
        # 时间序列预测模型
        # ------------------------------
        # IMS: Iterated Multi-Step forecating
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()  # 季节组分
            self.Linear_Trend = nn.ModuleList()  # 趋势组分
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        # DMS: Direct Multi-Step forecating
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        # 时间序列分类处理
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # ------------------------------
        # 时间序列分解
        # ------------------------------
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        # ------------------------------
        # 时间序列预测
        # ------------------------------
        # IMS: Iterated Multi-Step forecating
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len], 
                dtype = seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len], 
                dtype = trend_init.dtype
            ).to(trend_init.device)
            
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        # DMS: Direct Multi-Step forecating
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        # ------------------------------
        # 时间序列分析输出
        # ------------------------------
        x = seasonal_output + trend_output

        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        """
        预测
        """
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        """
        缺失填充
        """
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """
        异常检测
        """
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """
        分类
        """
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        output = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]

        return None




# 测试代码 main 函数
def main():
    # args
    class Config:
        task_name = "long_term_forecast"
        seq_len = 96
        pred_len = 96
        enc_in = 7
        moving_avg = 25
    configs = Config()
    
    # model
    model = Model(configs, individual = False)
    print(model)

if __name__ == "__main__":
    main()
