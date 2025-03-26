# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
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

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.log_util import logger

plt.switch_backend('agg')

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def adjust_learning_rate(optimizer, epoch, args):
    """
    学习率调整

    Args:
        optimizer (_type_): _description_
        epoch (_type_): _description_
        args (_type_): _description_
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f'Epoch: {epoch}, \tUpdating learning rate to {lr}')


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, epoch, model, optimizer=None, scheduler=None, model_path=""):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'Epoch: {epoch+1}, \tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model, optimizer=None, scheduler=None, model_path: str=""):
        # 日志打印
        if self.verbose:
            logger.info(f'Epoch: {epoch+1}, \tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # 模型保存
        training_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            # "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
        self.val_loss_min = val_loss


class StandardScaler():
    
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
