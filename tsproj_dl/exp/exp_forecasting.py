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

from loguru import logger
import numpy as np
from numpy import newaxis
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensorflow import keras
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential, load_model

from tsproj_dl.exp.exp_basic import Exp_Basic
from tsproj_dl.models.GRU import GRU
from tsproj_dl.config.gru import Config
from tsproj_dl.config.lstm import (
    Config_Univariate_SingleOutput_V1,
    Config_Univariate_SingleOutput_V2,
    Config_MultiVariate_SingleOutput,
    Config_MultiVariate_MultiOutput,
)
from timer import Timer
from tsproj_dl.data_provider.data_loader import Data_Loader
from tsproj_dl.utils.plot_results import plot_results, plot_results_multiple

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


config_dict = {
    "GRU": Config,
    "Univariate_SingleOutput_V1": Config_Univariate_SingleOutput_V1,
    "Univariate_SingleOutput_V2": Config_Univariate_SingleOutput_V2,
    "MultiVariate_SingleOutput": Config_MultiVariate_SingleOutput,
    "MultiVariate_MultiOutput": Config_MultiVariate_MultiOutput,
}


class Exp_Forecasting(Exp_Basic):
    """
    LSTM 模型
    """

    def __init__(self, args):
        super(Exp_Forecasting, self).__init__(args)
        self.config = config_dict[self.args.model_name]()

    def load_model(self, weights: str = False):
        logger.info(f'[Model] Loading model from file {self.config.save_path}')
        if weights:
            self.model = self.model_dict[self.config.model_name]
            self.model.load_state_dict(torch.load(self.config.save_path))
        else:
            self.model = torch.load(self.config.save_path)
        self.eval()

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        logger.info('[Model] Model Builded.')
        
        return model
    
    def _get_data(self):
        data_loader = Data_Loader(cfgs = self.config)
        train_loader, test_loader = data_loader.run()
        
        return data_loader, train_loader, test_loader
    
    def _select_criterion(self, loss_name = "MSE"):
        """
        评价指标
        """
        # if loss_name == "MSE":
        #     return nn.MSELoss()
        # elif loss_name == "MAPE":
        #     return mape_loss()
        # elif loss_name == "MASE":
        #     return mase_loss()
        # elif loss_name == "SMAPE":
        #     return smape_loss()
        self.loss_function = nn.MSELoss()
    
    def _select_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            params = self.model.parameters(), 
            lr = self.config.learning_rate
        )

    def train(self):
        """
        模型训练
        """
        # mark time
        timer = Timer()
        timer.start()
        
        # data load
        data_loader, train_loader, test_loader = self._get_data()
        
        logger.info('[Model] Training Started')
        # TODO 回调函数
        # save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
        # callbacks = [
        #     EarlyStopping(monitor = 'val_loss', patience = 2),
        #     ModelCheckpoint(filepath = save_fname, monitor = 'val_loss', save_best_only = True)
        # ]
        
        # 模型训练
        for epoch in range(self.config.epochs):
            # ------------------------------
            # model training
            # ------------------------------
            self.model.train()
            running_loss = 0
            train_bar = tqdm(train_loader)
            for data in train_bar:
                # batch data
                x_train, y_train = data
                # clear grad
                self.optimizer.zero_grad()
                # forward
                y_train_pred = self.model(x_train)
                loss = self.loss_function(y_train_pred, y_train.reshape(-1, 1))
                # backward
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch+1}/{self.config.epochs:.3f}] loss:{loss:.3f}"
            # ------------------------------
            # model validation
            # ------------------------------
            # 模型验证
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                test_bar = tqdm(test_loader)
                for data in test_bar:
                    # batch data
                    x_test, y_test = data
                    # forward
                    self.y_test_pred = self.model(x_test)
                    test_loss = self.loss_function(self.y_test_pred, y_test.reshap(-1, 1))
            if test_loss < self.config.best_loss:
                self.config.best_loss = test_loss
                torch.save(self.model.state_dict(), self.config.save_path)
                logger.info(f"model saved in {self.config.save_path}")
        logger.info("[Model] Finished Training")

        # mark time
        timer.stop()

    def predict(self, plot_size):
        # data load
        data_loader, train_loader, test_loader = self._get_data()
        # train result
        y_train_pred = data_loader.scaler.inverse_transform((self.model(data_loader.x_train_tensor).detach().numpy()[:plot_size]).reshape(-1, 1))
        y_train_true = data_loader.scaler.inverse_transform(data_loader.y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size])
        # test result
        y_test_pred = self.model(data_loader.x_test_tensor)
        y_test_pred = data_loader.scaler.inverse_transform(y_test_pred.detach().numpy()[:plot_size])
        y_test_true = data_loader.scaler.inverse_transform(data_loader.y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size])

        return (y_train_pred, y_train_true), (y_test_pred, y_test_true)

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        # mark time
        timer = Timer()
        timer.start()
        
        logger.info('[Model] Training Started')
        logger.info(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')
        # 回调函数
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
        callbacks = [
            ModelCheckpoint(filepath = save_fname, monitor = 'loss', save_best_only = True)
        ]
        # 模型拟合
        self.model.fit_generator(
            data_gen,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            callbacks = callbacks,
            workers = 1
        )
        logger.info(f'[Model] Training Completed. Model saved as {save_fname}')
        
        # mark time
        timer.stop()

    def save_model(self, weights: bool = False):
        """
        模型保存
        """
        logger.info(f'[Model] Training Completed. Model saved as {self.config.save_path}')
        if weights:
            # model weights
            torch.save(self.model.state_dict(), self.config.save_path)
        else:
            # whole model
            torch.save(self.model, self.config.save_path)
 
    def predict_direct_multi_output(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
        """
        多步预测

        Args:
            x_train_tensor (_type_): _description_
            y_train_tensor (_type_): _description_
            x_test_tensor (_type_): _description_
            y_test_tensor (_type_): _description_
        """
        plot_size = 200
        train_preds = self.model(x_train_tensor).detach().numpy()[:plot_size]
        train_true = y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size]
        test_preds = self.model(x_test_tensor).detach().numpy()[:plot_size]
        test_ture = y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size]

    def predict_recursive_multi_step(self, data):
        pass
    
    def predict_direct_multi_step(self, data):
        pass
    
    def predict_recursive_hybird(self, data):
        pass

    def predict_seq2seq_multi_step(self, data):
        pass
    
    def predict_point_by_point(self, data):
        logger.info('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size: int, prediction_len: int):
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
        prediction_seqs = []  # (20, 50, 1)
        for i in range(int(len(data) / prediction_len)):  # 951 / 50 = 19
            curr_frame = data[i * prediction_len]  # (49, 1)
            predicted = []  # 50
            for j in range(prediction_len):  # 50
                preds = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]  # curr_frame[newaxis, :, :].shape: (1, 49, 1) => (1,)
                predicted.append(preds)
                curr_frame = curr_frame[1:]  # (48, 1)
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis = 0)
            prediction_seqs.append(predicted)
        return prediction_seqs

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
        predicted = []
        for i in range(len(data)):
            preds = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]
            predicted.append(preds)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis = 0)
        return predicted

    @staticmethod
    def plot_train_results(pred, true):
        plt.figure(figsize = (12, 8))
        plt.plot(pred, "b", label = "Pred")
        plt.plot(true, "r", label = "True")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show();


class Trainer:
    """
    LSTM 模型
    """

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        logger.info(f'[Model] Loading model from file {filepath}')
        self.model = load_model(filepath)

    def build_model(self, configs):
        """
        模型构建
        """
        # mark time
        timer = Timer()
        timer.start()
        # 模型构建
        for layer in configs['model']['layers']:
            # 网络参数
            neurons = layer['neurons'] if 'neurons' in layer else None  # 神经元数量
            dropout_rate = layer['rate'] if 'rate' in layer else None  # dropout 概率
            activation = layer['activation'] if 'activation' in layer else None  # 激活函数
            return_seq = layer['return_seq'] if 'return_seq' in layer else None  # 是否返回序列
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None  # 输入时序长度
            input_dim = layer['input_dim'] if 'input_dim' in layer else None  # 输入时序维度，1:单变量序列；n:多变量序列
            # 构建网络
            if layer['type'] == 'dense':  # 全连接层
                self.model.add(
                    Dense(units = neurons, activation = activation)
                )
            if layer['type'] == 'lstm':  # LSTM 层
                self.model.add(
                    LSTM(
                        units = neurons, 
                        input_shape = (input_timesteps, input_dim), 
                        return_sequences = return_seq
                    )
                )
            if layer['type'] == 'dropout':  # Dropout 层
                self.model.add(
                    Dropout(dropout_rate)
                )
        # 模型编译
        self.model.compile(
            loss = configs['model']['loss'], 
            optimizer = configs['model']['optimizer'],
            metrics = None,
        )
        logger.info('[Model] Model Compiled')
        # mark time
        timer.stop()
        return self.model

    def train(self, x, y, epochs, batch_size, save_dir):
        """
        模型训练

        Args:
            x (_type_): _description_
            y (_type_): _description_
            epochs (_type_): 训练轮数
            batch_size (_type_): batch size
            save_dir (_type_): 模型保存路径
        """
        # mark time
        timer = Timer()
        timer.start()
        logger.info('[Model] Training Started')
        logger.info(f'[Model] {epochs} epochs, {batch_size} batch size')
        # 回调函数
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
        callbacks = [
            EarlyStopping(monitor = 'val_loss', patience = 2),
            ModelCheckpoint(filepath = save_fname, monitor = 'val_loss', save_best_only = True)
        ]
        # 模型拟合
        self.model.fit(
            x, 
            y,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )
        self.model.save(save_fname)
        logger.info(f'[Model] Training Completed. Model saved as {save_fname}')
        # mark time
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        # mark time
        timer = Timer()
        timer.start()
        logger.info('[Model] Training Started')
        logger.info(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')
        # 回调函数
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
        callbacks = [
            ModelCheckpoint(filepath = save_fname, monitor = 'loss', save_best_only = True)
        ]
        # 模型拟合
        self.model.fit_generator(
            data_gen,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            callbacks = callbacks,
            workers = 1
        )
        logger.info(f'[Model] Training Completed. Model saved as {save_fname}')
        # mark time
        timer.stop()


    def predict_point_by_point(self, data):
        """
        # TODO

        Args:
            data (_type_): 测试数据

        Returns:
            _type_: 预测序列
        """
        logger.info('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size: int, prediction_len: int):
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
        prediction_seqs = []  # (20, 50, 1)
        for i in range(int(len(data) / prediction_len)):  # 951 / 50 = 19
            curr_frame = data[i * prediction_len]  # (49, 1)
            predicted = []  # 50
            for j in range(prediction_len):  # 50
                preds = self.model.predict(curr_frame[newaxis, :, :])[0, 0]  # curr_frame[newaxis, :, :].shape: (1, 49, 1) => (1,)
                predicted.append(preds)
                curr_frame = curr_frame[1:]  # (48, 1)
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis = 0)
            prediction_seqs.append(predicted)
        return prediction_seqs

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
        predicted = []
        for i in range(len(data)):
            preds = self.model.predict(curr_frame[newaxis, :, :])[0, 0]
            predicted.append(preds)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis = 0)
        return predicted




# 测试代码 main 函数
def main():
    from tsproj_dl.config.gru import Config
    from tsproj_dl.data_provider.data_loader import Data_Loader

    # config
    config = Config()
    
    # data
    data_loader = Data_Loader(cfgs = config)
    train_loader, test_loader = data_loader.run()
    
    # model
    model = GRU(cfgs = config)
    
    # loss
    loss_func = nn.MSELoss()
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    
    # model train
    (y_train_pred, y_train_true), (y_test_pred, y_test_true) = train(
        config = config,
        train_loader = train_loader,
        test_loader = test_loader,
        model = model,
        loss_func = loss_func,
        optimizer = optimizer,
        x_train_tensor = data_loader.x_train_tensor, 
        y_train_tensor = data_loader.y_train_tensor,
        x_test_tensor = data_loader.x_test_tensor,
        y_test_tensor = data_loader.y_test_tensor,
        plot_size = 200,
        scaler = data_loader.scaler,
    )
    print(y_train_pred)
    print(y_test_pred)
    # result plot
    plot_train_results(y_train_pred, y_train_true)
    plot_train_results(y_test_pred, y_test_true)
    
    # ------------------------------
    # test
    # ------------------------------
    # model = nn.GRU(input_size=3, hidden_size=10, num_layers=2, bias=True, batch_first=True, bidirectional=False)
    # x = torch.randn(1, 5, 3)
    # output, h_0 = model(x)
    # print(output.shape)
    # print(h_0.shape)

    # ------------------------------
    # TODO
    # ------------------------------
    """
    # 数据名称
    data_name = "sinewave"
    
    # 读取配置文件
    configs = load_config(f"config_{data_name}.json")
    
    # ------------------------------
    # 读取数据
    # ------------------------------
    data = Data_Loader(
        filename = os.path.join('data', configs['data']['filename']),
        split_ratio = configs['data']['train_test_split'],
        cols = configs['data']['columns'],
    )
    # 训练数据
    x, y = data.get_train_data(
        seq_len = configs['data']['sequence_length'],
        normalise = configs['data']['normalise']
    )
    logger.info(f"x shape={x.shape}")
    logger.info(f"y shape={y.shape}")
    # 测试数据
    x_test, y_test = data.get_test_data(
        seq_len = configs['data']['sequence_length'],
        normalise = configs['data']['normalise']
    )
    logger.info(f"x_test shape={x_test.shape}")
    logger.info(f"y_test shape={y_test.shape}")
    # ------------------------------
    # 模型
    # ------------------------------
    # 创建 RNN 模型
    model = Trainer()
    mymodel = model.build_model(configs)
    plot_model(mymodel, to_file = configs["model"]["save_img"], show_shapes = True)
    # ------------------------------
    # 训练模型
    # ------------------------------
    # 模型训练
    # model.train(
    #     x,
    #     y,
    #     epochs = configs['training']['epochs'],
    #     batch_size = configs['training']['batch_size'],
    #     save_dir = configs['model']['save_dir']
    # )
    # 模型加载
    model.load_model(filepath = f'{configs["model"]["save_dir"]}/23052023-230644-e2.h5')
    # ------------------------------
    # 模型测试
    # ------------------------------
    # multi-sequence
    # --------------
    predictions_multiseq = model.predict_sequences_multiple(
        data = x_test, # shape: (656, 49, 1)
        window_size = configs['data']['sequence_length'],  # 50
        prediction_len = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(predictions_multiseq).shape)
    # plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'], title = data_name)
    
    # point by point
    # --------------
    predictions_pointbypoint = model.predict_point_by_point(data = x_test)
    logger.info(np.array(predictions_pointbypoint).shape)
    # plot_results(predictions_pointbypoint, y_test, title = data_name)
    
    # full-sequence
    # --------------
    prediction_fullseq = model.predict_sequence_full(
        data = x_test,
        window_size = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(prediction_fullseq).shape)
    # plot_results(prediction_fullseq, y_test, title = data_name)
    """

if __name__ == "__main__":
    main()
