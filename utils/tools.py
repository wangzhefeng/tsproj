# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tools.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-04
# * Version     : 0.1.010414
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
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

plt.switch_backend('agg')

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def adjust_learning_rate(optimizer, epoch, args):
    """
    学习率调整

    Args:
        optimizer (_type_): 模型优化器
        epoch (_type_): 训练 epoch
        args (_type_): 参数集
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {
            epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 
            4: 1e-5, 
            6: 5e-6, 
            8: 1e-6,
            10: 5e-7, 
            15: 1e-7, 
            20: 5e-8,
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    # 更新 optimizer 的 learning rate
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    """
    早停机制
    checkpoint 保存
    """

    def __init__(self, patience = 7, verbose = False, delta = 0):
        self.patience = patience  # 早停步数忍耐度
        self.verbose = verbose  # 是否打印日志
        self.delta = delta  # 阈值

        self.best_score = None  # 最小验证损失记录
        self.counter = 0  # 早停步数
        self.early_stop = False  # 是否早停
        self.val_loss_min = np.Inf  # 最小验证损失

    def __call__(self, val_loss, model, path):
        score = -val_loss  # loss 递增
        # 更新最好的结果
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        checkpoint 保存

        Args:
            val_loss (_type_): 验证损失
            model (_type_): 模型
            path (_type_): checkpoint 保存路径
        """
        # 验证损失变化日志打印
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        # checkpoint 保存
        torch.save(model.state_dict(), os.path.join(path, "/checkpoint.pth"))
        # 更新最小验证损失
        self.val_loss_min = val_loss


def adjustment(gt, pred):
    """
    异常检测

    Args:
        gt (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
            
    return gt, pred




# 测试代码 main 函数
def main():
   class Config:
       train_epochs = 10
       learning_rate = 1e-3
       lradj = 'type1'

   config = Config()
   
   adjust_learning_rate(optimizer=None, epoch=1, args=config)
       
if __name__ == "__main__":
   main()
