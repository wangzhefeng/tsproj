# -*- coding: utf-8 -*-

# ***************************************************
# * File        : EarlyStopping.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-13
# * Version     : 0.1.051316
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

import numpy as np
import paddle

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class EarlyStopping:
    """
    早停

    当验证集超过 patience 个 epoch 没有出现更好的评估分数，及早终止训练
    若当前 epoch 表现超过历史最佳分数，保存该节点模型

    参考：https://blog.csdn.net/m0_63642362/article/details/121244655
    """
    
    def __init__(self, 
                 patience = 7, 
                 verbose = False, 
                 delta = 0, 
                 ckp_save_path = "models/model_checkpoint_windid_04.pdparams") -> None:
        self.patience = patience  # val_loss 不更新容忍 epoch 次数，超过 patience，则停止训练
        self.verbose = verbose  # 是否输出日志信息
        self.counter = 0  # 当前 epoch 的 val_loss 没有超过历史最佳分数的计数器
        self.best_score = None  # 最佳评估分数
        self.early_stop = False  # 是否进行早停
        self.val_loss_min = np.Inf  # 最小 val_loss
        self.delta = delta  # TODO
        self.ckp_save_path = ckp_save_path  # 模型节点保存路径

    def __call__(self, val_loss, model):
        print(f"val_loss={val_loss}")
        score = -val_loss
        if self.best_score is None:  # 首轮，直接更新 best_score 和保存节点模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:  # 若当前 epoch 表现没超过历史最佳分数，且累积发生次数超过 patience，早停
            self.counter += 1
            print(f"Early Stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 若当前 epoch 表现超过历史最佳分数，更新 best_socre，保存该节点模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        paddle.save(model.state_dict(), self.ckp_save_path)
        self.val_loss_min = val_loss




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
