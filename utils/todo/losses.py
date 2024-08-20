# -*- coding: utf-8 -*-

# ***************************************************
# * File        : losses.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041900
# * Description : Loss functions for PyTorch.
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import torch
import torch.nn as nn

import paddle

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0

    return result


class mape_loss(nn.Module):

    def __init__(self):
        super(mape_loss, self).__init__()

    @staticmethod
    def forward(insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: (batch, time)
        :param target: Target values. Shape: (batch, time)
        :param mask: 0/1 mask. Shape: (batch, time)
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):

    def __init__(self):
        super(smape_loss, self).__init__()

    @staticmethod
    def forward(insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: (batch, time)
        :param target: Target values. Shape: (batch, time)
        :param mask: 0/1 mask. Shape: (batch, time)
        :return: Loss value
        """
        return 200 * torch.mean(
            divide_no_nan(
                torch.abs(forecast - target), 
                torch.abs(forecast.data) + torch.abs(target.data)
            ) * mask
        )


class mase_loss(nn.Module):

    def __init__(self):
        super(mase_loss, self).__init__()

    @staticmethod
    def forward(insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: (batch, time_i)
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: (batch, time_o)
        :param target: Target values. Shape: (batch, time_o)
        :param mask: 0/1 mask. Shape: (batch, time_o)
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim = 1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


class MultiTaskMSELoss(paddle.nn.Layer):
    """
    设置损失函数, 多任务模型，两个任务MSE的均值做loss输出
    """
    
    def __init__(self):
        super(MultiTaskMSELoss, self).__init__()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
