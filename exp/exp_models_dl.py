# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_csdn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-06-10
# * Version     : 0.1.061019
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
from typing import Dict

from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def train(config: Dict, 
          train_loader, test_loader, 
          model, loss_func, optimizer, 
          x_train_tensor,
          y_train_tensor,
          x_test_tensor,
          y_test_tensor,
          plot_size = 200,
          scaler = None):
    """
    模型训练
    """
    for epoch in range(config.epochs):
        # ------------------------------
        # model train
        # ------------------------------
        model.train()
        running_loss = 0
        train_bar = tqdm(train_loader)
        for data in train_bar:
            # batch data
            x_train, y_train = data
            # clear grad
            optimizer.zero_grad()
            # forward
            y_train_pred = model(x_train)
            loss = loss_func(y_train_pred, y_train.reshape(-1, 1))
            # backward
            loss.backward()
            optimizer.step()
            # loss cumsum
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{config.epochs:.3f} loss:{loss}]"
        # ------------------------------
        # model validate
        # ------------------------------
        model.eval()
        test_loss = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for data in test_bar:
                # batch data
                x_test, y_test = data
                # forward
                y_test_pred = model(x_test)
                test_loss = loss_func(y_test_pred, y_test.reshape(-1, 1))
        print("Finished Training.")
        # ------------------------------
        # 保存模型
        # ------------------------------
        print(test_loss)
        if test_loss < config.best_loss:
            config.best_loss = test_loss
            torch.save(model.state_dict(), config.save_path)
            logger.info(f"model saved in {config.save_path}")
    # ------------------------------
    # result inverse transform 
    # ------------------------------  
    # train result
    y_train_pred = scaler.inverse_transform((model(x_train_tensor).detach().numpy()[:plot_size]).reshape(-1, 1))
    y_train_true = scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size])
    # test result
    y_test_pred = model(x_test_tensor)
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy()[:plot_size])
    y_test_true = scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size])

    return (y_train_pred, y_train_true), (y_test_pred, y_test_true)


def plot_train_results(pred, true):
    plt.figure(figsize = (12, 8))
    plt.plot(pred, "b")
    plt.plot(true, "r")
    # plt.legend()
    plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
