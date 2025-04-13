# -*- coding: utf-8 -*-

# ***************************************************
# * File        : masking.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-04
# * Version     : 0.1.010415
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

import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TriangularCausalMask:

    def __init__(self, B, L, device="cpu"):
        """
        triangular causal mask

        Args:
            B (_type_): batch size
            L (_type_): sequence length
            device (str, optional): device. Defaults to "cpu".
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(diagonal=1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, 
            :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask



# 测试代码 main 函数
def main():
    # test TriangularCausalMask
    tri_mask = TriangularCausalMask(2, 10)
    mask_matrix = tri_mask._mask
    # print(mask_matrix.size())
    # print(mask_matrix)

    # test ProbMask
    prob_mask = ProbMask(1, 3, 5, torch.arange(5), torch.rand(1, 3, 5, 5))
    mask_matrix = prob_mask._mask
    # print(mask_matrix.size())
    # print(mask_matrix)

if __name__ == "__main__":
   main()
