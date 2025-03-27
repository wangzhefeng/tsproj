# -*- coding: utf-8 -*-

# ***************************************************
# * File        : todo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-27
# * Version     : 1.0.032711
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
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:
    
    # TODO 
    def _train_model_save(self, epoch, model, optimizer=None, scheduler=None, model_path=None):
        """
        模型保存
        """
        training_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
    
    # TODO
    def _train_model_load(self, model, optmizer=None, scheduler=None, model_path=None):
        """
        模型加载
        """
        checkpoints = torch.load(model_path)
        epoch = checkpoints["epoch"]
        if model:
            model.load_state_dict(checkpoints["model"])
        if optmizer:
            optmizer.load_state_dict(checkpoints["optmizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoints["scheduler"])

        return {
            "epoch": epoch, 
            "model": model, 
            "optimizer": optmizer, 
            "scheduler": scheduler
        }
    
    # TODO
    def train_recover(self, setting, log_dir):
        # 模型加载
        if os.path.exists(log_dir):
            load_input = self._train_model_load()
            start_epoch = load_input["epoch"]
            logger.info(f"加载 Epoch: {load_input['epoch']} 成功")
        else:
            start_epoch = 0
            logger.info(f"无保存模型，将从头开始训练...")
        
        for epoch in range(start_epoch+1, self.args.train_epochs):
            self.train(setting)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
