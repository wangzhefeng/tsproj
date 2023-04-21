# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-11
# * Version     : 0.1.041123
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from typing import List, Optional, Callable, Iterable

import torch
from torch import nn
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
import pytorch_lightning as pl


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def mean_abs_scaling(context, min_scale = 1e-5):
    return context.abs().mean().clamp(min_scale, None).unsqueeze(1)


class FeedForwardNetwork(nn.Module):

    def __init__(self, prediction_length: int, context_length: int, 
                 hidden_dimensions: List[int], batch_norm: bool = False,
                 distr_output: Callable = StudentTOutput(),  
                 scaling: Callable = mean_abs_scaling) -> None:
        super(FeedForwardNetwork, self).__init__()
        # ------------------------------
        # Parameters
        # ------------------------------
        # check params
        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0
        # params init
        self.prediction_length = prediction_length  # 预测长度
        self.context_length = context_length  # 10
        self.hidden_dimensions = hidden_dimensions  # 隐藏层维度 []
        self.distr_output = distr_output  # 分布输出
        self.batch_norm = batch_norm  # 是否进行 BatchNormalization
        # ------------------------------
        # Layers
        # ------------------------------
        # layer1: 数据转换
        self.scaling = scaling
        # layer2:
        modules = []
        dimensions = [context_length] + hidden_dimensions[:-1] # dimensions=[0, 1, 2, ..., n]
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            # layer2.1
            modules += [
                self.__make_linear(in_size, out_size), 
                nn.ReLU()
            ]
            # layer2.2
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        # layer3:
        modules.append(
            self.__make_linear(dimensions[-1], prediction_length * hidden_dimensions[-1])
        )
        # layer4: output        
        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])

    @staticmethod
    def __make_linear(dim_in, dim_out):
        linear = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(linear.weight, -0.07, 0.07)
        torch.nn.init.zeros_(linear.bias)
        return linear
    
    def forward(self, context):
        # data scaling
        scale = self.scaling(context)
        scaled_context = context / scale
        # output
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(-1, self.prediction_length, self.hidden_dimensions[-1])
        # student t distribution outout
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale

    def get_predictor(self, input_transform, batch_size = 32, device = None):
        return PyTorchPredictor(
            prediction_length = self.prediction_length,
            input_names = ["past_target"],
            prediction_net = self,
            batch_size = batch_size,
            input_transform = input_transform,
            forecast_generator = DistributionForecastGenerator(self.distr_output),
            device = device,
        )


class LightningFeedForwardNetwork(FeedForwardNetwork, pl.LightningModule):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # TODO
        context = batch["past_target"]
        target = batch["future_target"]
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        # loss function
        loss = -distr.log_prob(target)
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer




# 测试代码 main 函数
def main():
    from gluonts.dataset.repository.datasets import get_dataset
    dataset = get_dataset("electricity")

if __name__ == "__main__":
    main()
