# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 1.0.021317
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

import torch

from models import (
    DLinear, 
    Autoformer, 
    PatchTST,
    TimesNet, 
    Transformer,
    Transformer_original,
)
# from models_dl import (
#     MLP,
#     RNN,
#     GRU,
#     LSTM,
#     BiLSTM,
#     Attention,
#     CNN_Attention,
#     CNN_Conv1D,
#     CNN_Conv2D,
#     CNN_LSTM_Attention,
#     InformerTodo, 
#     Seq2Seq_LSTM,
#     LSTM_Attention,
#     LSTM_CNN,
#     TCN,
#     Transformer,
# )
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Basic:

    def __init__(self, args):
        # 参数
        self.args = args
        # 模型集
        self.model_dict = {
            # ------------------------------
            # Time Series Library models
            # ------------------------------
            "Transformer_original": Transformer_original,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            # 'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            # 'TimeMixer': TimeMixer,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            # "SCINet": SCINet,
            # 'PAttn': PAttn,
            # 'TimeXer': TimeXer,
            # 'WPMixer': WPMixer,
            # 'MultiPatchFormer': MultiPatchFormer
            # ------------------------------
            # Basic Neural Network model
            # ------------------------------
            # "MLP": MLP,
            # "RNN": RNN,
            # "GRU": GRU,
            # "LSTM": LSTM,
            # "BiLSTM": BiLSTM,
            # "Attention": Attention,
            # "CNN_Attention": CNN_Attention,
            # "CNN_Conv1D": CNN_Conv1D,
            # "CNN_Conv2D": CNN_Conv2D,
            # "CNN_LSTM_Attention": CNN_LSTM_Attention,
            # "InformerTodo": InformerTodo, 
            # "Seq2Seq_LSTM": Seq2Seq_LSTM,
            # "LSTM_Attention": LSTM_Attention,
            # "LSTM_CNN": LSTM_CNN,
            # "TCN": TCN,
            # "Transformer": Transformer,
        }
        if args.model == 'Mamba':
            logger.info('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict["Mamba"] = Mamba
        # 设备
        self.device = self._acquire_device()
        # 模型构建
        self.model = self._build_model().to(self.device)
    
    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda": 
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device(f'cuda:{self.args.gpu}')
            logger.info(f'Use GPU: cuda:{self.args.gpu}.')
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            logger.info(f"Use GPU: mps.")
        else:
            device = torch.device('cpu')
            logger.info('Use CPU.')

        return device
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def train(self):
        pass

    def vali(self):
        pass

    def test(self):
        pass





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
