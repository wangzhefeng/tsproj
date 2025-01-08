# -*- coding: utf-8 -*-

# ***************************************************
# * File        : conv1d_demo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-07
# * Version     : 1.0.010723
# * Description : https://blog.csdn.net/shebao3333/article/details/140632930
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TestConv1d_1(nn.Module):
    
    def __init__(self):
        super(TestConv1d_1, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = 1,
            out_channels = 1,
            kernel_size = 1,
            bias = False,
        )
        self.init_weights()
    
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        with torch.no_grad():
            self.conv.weight[0, 0, 0] = 2.0
    

class TestConv1d_2(nn.Module):
    
    def __init__(self):
        super(TestConv1d_2, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        with torch.no_grad():
            self.conv.weight[0, 0, 0] = 2.0
            self.conv.weight[0, 0, 1] = 2.0


class TestConv1d_3(nn.Module):
    
    def __init__(self):
        super(TestConv1d_3, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)
 
 
class TestConv1d_4(nn.Module):
    
    def __init__(self):
        super(TestConv1d_4, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)
 
 
class TestConv1d_5(nn.Module):
    
    def __init__(self):
        super(TestConv1d_5, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            stride=3, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)
 
 
class TestConv1d_6(nn.Module):
    
    def __init__(self):
        super(TestConv1d_6, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            dilation=2, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)
 
 
class TestConv1d_7(nn.Module):
     
    def __init__(self):
        super(TestConv1d_7, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=2, 
            out_channels=2, 
            kernel_size=1, 
            groups=2, 
            bias=False
        )
        self.init_weights()
 
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        print(self.conv.weight.shape)
        self.conv.weight[0,0,0] = 2.
        self.conv.weight[1,0,0] = 4.
 



# 测试代码 main 函数
def main():
    in_x = torch.tensor([[[1, 2, 3, 3, 4, 5, 6]]]).float()
    print("in_x.shape", in_x.shape)
    print(in_x)
    net = TestConv1d_3()
    out_y = net(in_x)
    print("out_y.shape", out_y.shape)
    print(out_y)

if __name__ == "__main__":
    main()
