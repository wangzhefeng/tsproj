# -*- coding: utf-8 -*-

# ***************************************************
# * File        : arima.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-17
# * Version     : 0.1.051722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from random import random
import joblib
import datetime as dt

from utils.log_util import logger
import numpy as np
from numpy import newaxis
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from utils.timer import Timer

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def __getnewargs__(self):
    """
    Monkey patch(猴子补丁) around bug in ARIMA class

    因为在 statsmodels 中尚未定义 pickle 所需的函数，
    在保存模型之前，必须在 ARIMA 模型中定义 __getnewargs__ 函数，
    它定义构造对象所需的参数。
    解决这个问题。修复涉及两件事情：
    1. 定义适用于 ARIMA 对象的 __getnewargs__ 函数的实现
    2. 将新的函数添加到 ARIMA

    ref: 1.http://www.atyun.com/4346.html
         2.https://github.com/statsmodels/statsmodels/pull/3217
    """
    return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__


class Model:
    """
    AR 模型
    """
    
    def __init__(self) -> None:
        pass

    def load_model(self, filepath):
        """
        加载 .pkl 模型文件
        """
        logger.info(f"[Model] Loading model from file {filepath}")
        self.model = ARIMAResults.load(filepath)
    
    def build_model(self, config):
        self.model = ARIMA(x, order = (1, 1, 1))
    
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
    import matplotlib.pyplot as plt


if __name__ == "__main__":
    main()
