# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lstm_v2.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052710
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
import tqdm
import datetime as dt
from typing import Dict
# from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from keras.utils import plot_model
# from keras.layers import LSTM, Activation, Dense, Dropout
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import Sequential, load_model

from tsproj_dl.exp.exp_basic import Exp_Basic

from tsproj_dl.data_provider.data_loader import Data_Loader, Data_Loader_todo
from utils.losses import mape_loss, mase_loss, smape_loss
from tsproj_dl.utils.plot_results import plot_results, plot_results_multiple
from tsproj_dl.utils.timer import Timer
from tsproj_dl.utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Forecasting(Exp_Basic):
    """
    时间序列预测
    """

    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
        
        # self.model_name = self.args.model_name
        # self.use_gpu = self.args.use_gpu
        # self.gpu = self.args.gpu
        # self.devices = self.args.devices
        # self.use_multi_gpu = self.args.use_multi_gpu
        # self.device_ids = self.args.device_ids
        # self.learning_rate = self.args.learning_rate
        # self.loss_name = self.args.loss_name
        # self.epochs = self.args.epochs
        # self.best_loss = self.args.best_loss
        # self.save_path = self.args.save_path
    
    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        logger.info('[Model] Model Builded.')
        
        return model
    
    def _get_data(self):
        data_loader = Data_Loader(cfgs = self.args)
        train_loader, test_loader = data_loader.run()
        
        return data_loader, train_loader, test_loader
    
    def _select_criterion(self, loss_name = "MSE"):
        """
        评价指标
        """
        if loss_name == "MSE":
            return nn.MSELoss()
        elif loss_name == "MAPE":
            return mape_loss()
        elif loss_name == "MASE":
            return mase_loss()
        elif loss_name == "SMAPE":
            return smape_loss()
    
    def _select_optimizer(self):
        model_optim = torch.optim.AdamW(
            params = self.model.parameters(), 
            lr = self.args.learning_rate
        )
        
        return model_optim

    def train(self):
        """
        模型训练
        """
        # mark time
        timer = Timer()
        timer.start()
        
        # data load
        _, train_loader, test_loader = self._get_data()
        # loss func
        criterion = self._select_criterion(self.args.loss_name)
        # optimizer
        model_optim = self._select_optimizer()
        
        logger.info('[Model] Training Started')        
        # 模型训练
        for epoch in range(self.args.epochs):
            # ------------------------------
            # model training
            # ------------------------------
            self.model.train()
            
            running_loss = 0
            # TODO train_bar = tqdm(train_loader)
            # TODO for data in train_bar:
            for data in train_loader:
                # batch data
                x_train, y_train = data
                # clear grad
                model_optim.zero_grad()
                # forward
                y_train_pred = self.model(x_train)
                loss = criterion(y_train_pred, y_train.reshape(-1, 1))
                # backward
                loss.backward()
                model_optim.step()
                running_loss += loss.item()
                # TODO train_bar.desc = f"train epoch[{epoch+1}/{self.args.epochs:.3f}] loss:{loss:.3f}"
            # ------------------------------
            # model validation
            # ------------------------------
            # 模型验证
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                # TODO test_bar = tqdm(test_loader)
                # TODO for data in test_bar:
                for data in test_loader:
                    # batch data
                    x_test, y_test = data
                    # forward
                    self.y_test_pred = self.model(x_test)
                    test_loss = criterion(self.y_test_pred, y_test.reshape(-1, 1))
            if test_loss < self.args.best_loss:
                self.args.best_loss = test_loss
                torch.save(self.model.state_dict(), self.args.save_path)
                logger.info(f"model saved in {self.args.save_path}")
        logger.info("[Model] Finished Training")

        # mark time
        timer.stop()
    
    def save_model(self, weights: bool = False):
        """
        模型保存
        """
        logger.info(f'[Model] Training Completed. Model saved as {self.args.save_path}')
        if weights:
            torch.save(self.model.state_dict(), self.args.save_path)  # model weights
        else:
            torch.save(self.model, self.args.save_path)  # whole model

    def load_model(self, weights: str = False):
        logger.info(f'[Model] Loading model from file {self.args.save_path}')
        if weights:
            self.model = self.model_dict[self.args.model]
            self.model.load_state_dict(torch.load(self.args.save_path))
        else:
            self.model = torch.load(self.args.save_path)
        self.eval()

    def predict_single_step(self, data):
        """
        单步预测

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        logger.info('[Model] Predicting Point-by-Point...')
        pred = self.model.predict(data)
        pred = np.reshape(pred, (pred.size,))
        
        return pred

    def predict_directly_multi_output(self, plot_size):
        # data load
        data_loader, _, _ = self._get_data()
        # train result
        y_train_pred = data_loader.scaler.inverse_transform(
            (self.model(data_loader.x_train_tensor).detach().numpy()[:plot_size]).reshape(-1, 1)
        )
        y_train_true = data_loader.scaler.inverse_transform(
            data_loader.y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size]
        )
        # test result
        y_test_pred = data_loader.scaler.inverse_transform(
            self.model(data_loader.x_test_tensor).detach().numpy()[:plot_size]
        )
        y_test_true = data_loader.scaler.inverse_transform(
            data_loader.y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size]
        )

        return (y_train_pred, y_train_true), (y_test_pred, y_test_true)

    def predict_recursive_multi_step(self, data, window_size: int, horizon: int):
        """
        时序多步预测
            - 每次预测使用 window_size 个历史数据进行预测，预测未来 prediction_len 个预测值
            - 每一步预测一个点，然后下一步将预测的点作为历史数据进行下一次预测

        Args:
            data (_type_): 测试数据
            window_size (int): 窗口长度
            prediction_len (int): 预测序列长度

        Returns:
            _type_: 预测序列
        """
        logger.info('[Model] Predicting Sequences Multiple...')
        preds_seq = []  # (20, 50, 1)
        for i in range(int(len(data) / horizon)):  # 951 / 50 = 19
            curr_frame = data[i * horizon]  # (49, 1)
            preds = []  # 50
            for j in range(horizon):  # 50
                pred = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]  # curr_frame[newaxis, :, :].shape: (1, 49, 1) => (1,)
                preds.append(pred)
                curr_frame = curr_frame[1:]  # (48, 1)
                curr_frame = np.insert(curr_frame, [window_size - 2], preds[-1], axis = 0)
            preds_seq.append(preds)
        
        return preds_seq
    
    def predict_direct_multi_step(self, data):
        pass
    
    def predict_recursive_hybird(self, data):
        pass

    def predict_seq2seq_multi_step(self, data):
        pass
    
    def predict_sequence_full(self, data, window_size: int):
        """
        单步预测

        Args:
            data (_type_): 测试数据
            window_size (_type_): 窗口长度

        Returns:
            _type_: 预测序列
        """
        logger.info('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        preds_seq = []
        for i in range(len(data)):
            pred = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]
            preds_seq.append(pred)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], preds_seq[-1], axis = 0)
        
        return preds_seq

    @staticmethod
    def plot_train_results(pred, true, task = "Train"):
        plt.figure(figsize = (15, 8))
        plt.plot(pred, "b", label = "Pred")
        plt.plot(true, "r", label = "True")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"{task} Predictions")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show();




# 测试代码 main 函数
def main():
    from tsproj_dl.config.gru import Config
    # config
    configs = Config()
    exp = Exp_Forecasting(args=configs)
    # model train
    exp.train()
    # model predict
    (y_train_pred, y_train_true), (y_test_pred, y_test_true) = exp.predict_directly_multi_output(plot_size = 200)
    logger.info(f"y_train_pred: \n{y_train_pred}")
    logger.info(f"y_test_pred: \n{y_test_pred}")
    # result plot
    exp.plot_train_results(y_train_pred, y_train_true, task="Train")
    exp.plot_train_results(y_test_pred, y_test_true, task="Test")
    '''
    # ------------------------------
    # 模型测试
    # ------------------------------
    model = None
    # multi-sequence
    # --------------
    predictions_multiseq = model.predict_sequences_multiple(
        data = x_test, # shape: (656, 49, 1)
        window_size = configs['data']['sequence_length'],  # 50
        prediction_len = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(predictions_multiseq).shape)
    plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'], title = configs.data)
    
    # point by point
    # --------------
    predictions_pointbypoint = model.predict_point_by_point(data = x_test)
    logger.info(np.array(predictions_pointbypoint).shape)
    plot_results(predictions_pointbypoint, y_test, title = configs.data)
    
    # full-sequence
    # --------------
    prediction_fullseq = model.predict_sequence_full(
        data = x_test,
        window_size = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(prediction_fullseq).shape)
    plot_results(prediction_fullseq, y_test, title = configs.data)
    '''
    
if __name__ == "__main__":
    main()
