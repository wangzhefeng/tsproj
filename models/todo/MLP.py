# -*- coding: utf-8 -*-


# ***************************************************
# * File        : MLP.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-16
# * Version     : 0.1.031614
# * Description : description
# * Link        : link
# * Requirement : https://mp.weixin.qq.com/s?__biz=MzkzMTMyMDQ0Mw==&mid=2247485132&idx=2&sn=70547a02346b21bd1682733d6fb8ee2e&chksm=c26d81d8f51a08ced66dcbd612e3c77ea75c7308a5457dc5d7bf3ea45fa551bdb868316f7344&scene=132#wechat_redirect
# ***************************************************


# python libraries
import os
import sys

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self,
                 d_in: int,
                 d_hid: int,
                 d_out: int,
                 num_layers: int,
                 activation: str = "relu"):
        super(MLP).__init__()
        self.num_layers = num_layers
        hidden_dims = [d_hid] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([d_in] + hidden_dims, hidden_dims + [d_out]))
        self.act = getattr(F, activation) if activation is not None else nn.Sequential()
        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = self.act(layer(x)) if idx < self.num_layers - 1 else layer(x)
        return x

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
