# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lstm.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052221
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
import datetime as dt

from loguru import logger
import numpy as np
from numpy import newaxis
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential, load_model

from utils.timer import Timer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
    pass

if __name__ == "__main__":
    main()
