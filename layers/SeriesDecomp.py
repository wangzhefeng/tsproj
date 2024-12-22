# -*- coding: utf-8 -*-

# ***************************************************
# * File        : SeriesDecomp.py
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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import List

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size = kernel_size, stride = stride, padding = 0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim = 1)
        
        # avgpool1d
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride = 1)

    def forward(self, x):
        # Trend
        moving_mean = self.moving_avg(x)
        # Seasonal
        res = x - moving_mean
        
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size: List[int]):
        super(series_decomp_multi, self).__init__()
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        res, moving_mean = [], []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)

        return sea, moving_mean




# 测试代码 main 函数
def main():
    x = torch.randn(1, 2, 15)
    print(x)
    print(x.size())

    # 移动平均
    # mv = moving_avg(kernel_size=3, stride=1)
    # x_mean = mv(x)
    # print(x_mean)
    
    # 时序分解
    decompsition = series_decomp(kernel_size = 3)
    seasonal, trend = decompsition(x)
    print(f"seasonal: \n{seasonal}")
    print(f"seasonal size: \n{seasonal.size()}")
    print(f"trend: \n{trend}")
    print(f"trend size: \n{trend.size()}")
    
if __name__ == "__main__":
    main()
