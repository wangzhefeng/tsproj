# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041902
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

from loguru import logger
import torch

from models import (
    Autoformer, 
    Crossformer, 
    DLinear, 
    ETSformer,
    FEDformer, 
    FiLM, 
    Informer, 
    InformerModel, 
    LightTS, 
    MICN, 
    Nonstationary_Transformer, 
    PatchTST, 
    Pyraformer, 
    Reformer,
    TimesNet, 
    Transformer
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Basic(object):

    def __init__(self, args):
        self.args = args
        # TODO
        self.model_dict = {
            'Autoformer': Autoformer,
            'Crossformer': Crossformer,
            'DLinear': DLinear,
            'ETSformer': ETSformer,
            'FEDformer': FEDformer,
            'FiLM': FiLM,
            'InformerRaw': Informer,
            'Informer': InformerModel,
            'LightTS': LightTS,
            'MICN': MICN,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'Reformer': Reformer,
            'TimesNet': TimesNet,
            'Transformer': Transformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.args.gpu}")
            logger.info(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            logger.info("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
