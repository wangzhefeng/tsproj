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

from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from models_dl.LSTM import LSTM
from models_dl.config.config_wind_lstm import (
    Config_Univariate_SingleOutput_V1,
    Config_Univariate_SingleOutput_V2,
    Config_MultiVariate_SingleOutput,
    Config_MultiVariate_MultiOutput,
)
from models_dl.utils.timer import Timer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model:
    """
    LSTM 模型
    """

    def __init__(self):
        self.config = Config_Univariate_SingleOutput_V1()

    def load_model(self, weights: str = False):
        logger.info(f'[Model] Loading model from file {self.config.save_path}')
        if weights:
            self.model = LSTM()
            self.model.load_state_dict(torch.load(self.config.save_path))
        else:
            self.model = torch.load(self.config.save_path)
        self.eval()

    def build_model(self):
        """
        模型构建
        """
        # mark time
        timer = Timer()
        timer.start()
        # model
        self.model = LSTM(
            feature_size = self.config.feature_size, 
            hidden_size = self.config.hidden_size, 
            num_layers = self.config.num_layers, 
            output_size = self.config.output_size,
        )
        self.loss_function = nn.MSELoss()  # TODO
        self.optimizer = torch.optim.AdamW(
            params = self.model.parameters(), 
            lr = self.config.learning_rate
        )
        logger.info('[Model] Model Builded.')
        # mark time
        timer.stop()
        return self.model

    def train(self, train_loader, test_loader, epochs, batch_size):
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
        # TODO 回调函数
        # save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
        # callbacks = [
        #     EarlyStopping(monitor = 'val_loss', patience = 2),
        #     ModelCheckpoint(filepath = save_fname, monitor = 'val_loss', save_best_only = True)
        # ]
        # 模型训练
        self.model = None
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0
            train_bar = tqdm(train_loader)
            for data in train_bar:
                x_train, y_train = data
                self.optimizer.zero_grad()
                y_train_pred = self.model(x_train)
                loss = self.loss_function(y_train_pred, y_train.reshape(-1, 1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch+1}/{self.config.epochs}] loss:{loss:.3f}"
            # 模型验证
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                test_bar = tqdm(test_loader)
                for data in test_bar:
                    x_test, y_test = data
                    y_test_pred = self.model(x_test)
                    test_loss = self.loss_function(y_test_pred, y_test.reshap(-1, 1))
            if test_loss < self.config.best_loss:
                self.config.best_loss = test_loss
                torch.save(self.model.state_dict(), self.config.save_path)
        logger.info("[Model] Finished Training")

        # mark time
        timer.stop()

    # def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
    #     # mark time
    #     timer = Timer()
    #     timer.start()
    #     logger.info('[Model] Training Started')
    #     logger.info(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')
    #     # 回调函数
    #     save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{str(epochs)}.h5')
    #     callbacks = [
    #         ModelCheckpoint(filepath = save_fname, monitor = 'loss', save_best_only = True)
    #     ]
    #     # 模型拟合
    #     self.model.fit_generator(
    #         data_gen,
    #         steps_per_epoch = steps_per_epoch,
    #         epochs = epochs,
    #         callbacks = callbacks,
    #         workers = 1
    #     )
    #     logger.info(f'[Model] Training Completed. Model saved as {save_fname}')
    #     # mark time
    #     timer.stop()

    # TODO
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

    # TODO 多步预测
    def predict_direct_multi_output(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
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
