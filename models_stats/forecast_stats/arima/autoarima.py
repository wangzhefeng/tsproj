# -*- coding: utf-8 -*-

# ***************************************************
# * File        : autoarima.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-25
# * Version     : 0.1.052520
# * Description : description
# * Link        : 1.读书数据、数据预处理
# *               2.查看数据模式：趋势、季节、周期、随机误差
# *               3.Augmented Dickey-Fuller(ADF) 平稳性检验
# *               4.数据处理、模型选择
# *               5.训练集、测试集分割
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import joblib

from utils.log_util import logger
import numpy as np
import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima.arima import ADFTest

from utils.timer import Timer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model:
    """
    AutoARIMA 模型
    """

    # TODO    
    def __init__(self, data, window_len: int, predict_len: int) -> None:
        self.data = data
        self.train_data = None
        self.test_data = None
        self.window_len = window_len
        self.predict_len = predict_len

    def load_model(self, filepath):
        """
        加载 .pkl 模型文件
        """
        logger.info(f"[Model] Loading model from file {filepath}")
        with open(filepath, "rb") as pkl:
            self.model = joblib.load(pkl)
   
    # TODO 
    def build_model(self, config):
        # mark time
        timer = Timer()
        timer.start()
        # 模型构建
        self.model = auto_arima(
            self.train_data, 
            start_p = 0, d = 1, start_q = 0, 
            max_p = 5, max_d = 5, max_q = 5, 
            start_P = 0, D = 1, start_Q = 0, 
            max_P = 5, max_D = 5, max_Q = 5, 
            m = 12, 
            seasonal = True, 
            error_action = 'warn',
            trace = True,
            supress_warnings = True,
            stepwise = True,
            random_state = 20,
            n_fits = 50
        )
        logger.info(self.model.summary())
        # mark time
        timer.stop()

        return self.model
    
    def train(self, save_dir):
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

        # 模型拟合
        model_fit = self.model.fit()
        # 模型保存
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-.pkl')
        model_fit.save(save_fname)
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
