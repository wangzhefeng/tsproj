import os

import torch
from models import (
    # TimesNet, 
    # Autoformer, 
    DLinear, 
    # FEDformer
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


class Exp_Basic:

    def __init__(self, args):
        # 参数
        self.args = args
        # 模型集
        self.model_dict = {
            # ------------------------------
            # Time Series Library models
            # ------------------------------
            # 'TimesNet': TimesNet,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            # 'PatchTST': PatchTST,
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
            # 'TemporalFusionTransformer': TemporalFusionTransformer
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
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict[Mamba] = Mamba
        # 设备
        self.device = self._acquire_device()
        # 模型构建
        self.model = self._build_model().to(self.device)
    
    def _acquire_device(self):
        if self.args.use_gpu: 
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            # device = torch.device('cuda')
            print(f'Use GPU: cuda:{self.args.gpu}.')
        else:
            device = torch.device('cpu')
            print('Use CPU.')

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
