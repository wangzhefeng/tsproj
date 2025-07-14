# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ml_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-04-08
# * Version     : 1.0.040809
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
import warnings
import pickle
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor

warnings.filterwarnings("ignore")
warnings.simplefilter(action = "ignore", category = FutureWarning)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_data():
    pass


def build_data(data, col_list: List[str], response_col: str, lags: int):
    """
    时间序列数据整理，构建时序预测模型数据集
    """
    # 筛选变量
    data = data[col_list]
    # 构建时序预测模型数据集
    col_shifted = []
    for col_name in col_list:
        data[f'{col_name}0'] = data[col_name].shift()
        col_shifted.append(f'{col_name}0')
        for i in range(1, lags):
            data[f"{col_name}{i}"] = data[f"{col_name}{i-1}"].shift()
            col_shifted.append(f"{col_name}{i}")
    # 筛选变量
    col_final = [response_col] + col_shifted
    training_data = data[col_final]
    # 重命名
    training_data.columns = ['goal'] + [f"a{i}" for i in range(1, len(col_final))]
    
    return training_data


def model_train_data(config,
                     task: str,
                     raw_data_name: str, 
                     training_data_path: str, 
                     training_data_file: str, 
                     predict_vars: List,
                     response_var: str,
                     lags: int = 5):
    """
    训练数据生成

    Args:
        raw_data_name (str): 原始下载数据名称
        training_data_path (str): 训练数据保存路径
        training_data_file (str): 训练数据保存名称
        predict_vars (List): 预测变量名列表
        response_var (str): 目标变量名
        lags (int): 数据 shift 次数
    """
    # 数据读取及预处理
    data = load_data(
        config = config,
        data_file_name = raw_data_name,
        time_col_name = "time",
        upper_name = True,
        columns_map = {},
        resample = False,
        add_date = False,
        cn_name = False,
        set_index = False,
        prefix = config.col_prefix,
    )
    # 训练数据构造
    training_data = build_data(
        data = data, 
        col_list = predict_vars,
        response_col = response_var,
        lags = lags,
    )
    # 训练数据保存
    output_data_path = f"{training_data_path}/{training_data_file}.csv"
    if not os.path.exists(output_data_path):
        training_data.to_csv(output_data_path, index = False)


def model_train(training_data_path: str,
                training_data_file: str,
                model_path: str, 
                model_name: str, 
                model_version: str):
    """
    模型训练

    Args:
        training_data_path (str): 训练数据文件所在路径
        model_file_path (str): 模型文件保存路径
        model_file_name (str): 模型文件名
        model_file_version (str): 模型文件版本
    """
    # ------------------------------
    # 训练数据读取及处理
    # ------------------------------
    # 训练数据读取
    data = pd.read_csv(f"{training_data_path}/{training_data_file}.csv", index_col = False)
    # 缺失值填充
    data.fillna(0, inplace = True)
    # 训练数据分割
    data_x = data[data.columns[1:]]
    data_y = data[data.columns[0]]
    # ------------------------------
    # 模型训练
    # ------------------------------
    model = ExtraTreeRegressor(random_state = 0)
    model = BaggingRegressor(model, random_state = 0).fit(data_x, data_y)
    # ------------------------------
    # 模型保存
    # ------------------------------
    with open(os.path.join(model_path, f'{model_name}_{model_version}.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    return model


def model_predict(params: Dict, 
                  model_path: str, 
                  model_name: str, 
                  model_version: str, 
                  predict_vars: List,
                  output_name: str,
                  local_test: bool = False,
                  lags: int = 5):
    """
    模型预测
    """
    # ------------------------------
    # 模型加载
    # ------------------------------
    if local_test:
        pre_trained_model = joblib.load(os.path.join(model_path, f"{model_name}_{model_version}.pkl"))
    else:
        pre_trained_model = joblib.load(f"{model_name}_{model_version}.pkl")
    # ------------------------------
    # 模型预测
    # ------------------------------
    # 取数据
    input_data = [params[col] for col in predict_vars]
    # 预测数据构建
    data_list = [col[i] for col in input_data for i in range(lags)]
    columns_num = len(input_data) * lags
    columns_list = [f"a{i}" for i in range(1, columns_num + 1)]
    df = pd.DataFrame(
        np.reshape(a = data_list, newshape = (1, columns_num)),
        columns = columns_list,
    )
    # ------------------------------
    # 模型预测结果
    # ------------------------------
    model_preds = pre_trained_model.predict(df)
    # 预测数据控制
    # ycmq = min(float(LTJMQLL[0]) + MinMQLLRange, ycmq)
    # ycmq = max(float(LTJMQLL[0]) - MinMQLLRange, ycmq)
    # 输出结果
    model_output = {
        output_name: model_preds[0],
    }
    
    return model_output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
