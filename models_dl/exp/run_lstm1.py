# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052220
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

from loguru import logger
import numpy as np
from tensorflow import keras
from keras.utils import plot_model

from models_dl.utils.json_config_loader import load_config
from data_provider.DataLoader import DataLoader
from experiments.train_lstm import Trainer
from ts_visual.plot_results import plot_results, plot_results_multiple

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 测试代码 main 函数
def main():
    # 数据名称
    data_name = "sinewave"
    
    # 读取配置文件
    configs = load_config(f"config_{data_name}.json")
    
    # ------------------------------
    # 读取数据
    # ------------------------------
    data = DataLoader(
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
    

if __name__ == '__main__':
    main()
