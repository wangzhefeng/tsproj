# -*- coding: utf-8 -*-

# ***************************************************
# * File        : inference.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-14
# * Version     : 1.0.021417
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

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# gpu setting
logger.info(f"GPU available: {torch.cuda.is_available()}")
logger.info(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"current GPU name: {torch.cuda.get_device_name()}")
    logger.info(f"current GPU id: {torch.cuda.current_device()}")
    torch.cuda.set_device(0)

# params
args = {}

# exp
exp = Exp_Long_Term_Forecast(args)

# model
model = exp.model

# load model
setting = "ETTh1_96_24_Autoformer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0"
path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
model.load_state_dict(torch.load(path))





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
