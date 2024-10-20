# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lstm.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-12-18
# * Version     : 0.1.121808
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Dropout
from keras.layers import RNN, SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping

np.random.seed(2022)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


class RNNModel:
    def __init__(self, 
                 look_back = 1, 
                 epochs_purchase = 20, 
                 epochs_redeem = 40, 
                 batch_size = 1, 
                 verbose = 2, 
                 patience = 10, 
                 store_reulst = False):
        pass

    def access_data(self, data_frame):
        pass

    def create_data_set(self, data_set):
        pass

    def rnn_model(self, train_x, train_y, epochs):
        pass

    def predict(self, model, data):
        pass

    def plot_show(self, predict):
        pass

    def run(self):
        pass



# 测试代码 main 函数
def main():
    initiation = RNNModel(
        look_back = 40, 
        epochs_purchase = 150,
        epochs_redeem = 230,
        batch_size = 16,
        verbose = 2,
        patience = 50,
        store_result = False,
    )
    initiation.run()

if __name__ == "__main__":
    main()

