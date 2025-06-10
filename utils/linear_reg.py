# -*- coding: utf-8 -*-

# ***************************************************
# * File        : reg_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-10
# * Version     : 1.0.061014
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 绘图风格
plt.style.use("seaborn-v0_8-whitegrid")  # "ggplot", "classic", "darkgrid"
# 用来正常显示中文标签
plt.rcParams["font.sans-serif"]=["SimHei"]  # 'Arial Unicode MS'
# 处理 matplotlib 字体问题
plt.rcParams["font.family"].append("SimHei")
# 用来显示负号
plt.rcParams["axes.unicode_minus"] = False
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def linear_model(train_df: pd.DataFrame):
    """
    线性回归拟合

    Args:
        train_df (pd.DataFrame): 训练数据
    """
    # 数据分割
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1:]
    logger.info(f"X_train: \n{X_train} \nX_train.shape: {X_train.shape}")
    logger.info(f"y_train: \n{y_train} \ny_train.shape: {y_train.shape}")
    # 标准化
    scale = StandardScaler()
    scale_y = StandardScaler()
    x_train_scaled = scale.fit_transform(X_train.values)
    y_train_scaled = scale_y.fit_transform(y_train.values)
    logger.info(f"x_train_scaled: \n{x_train_scaled} \nx_train_scaled.shape: {x_train_scaled.shape}")
    logger.info(f"y_train_scaled: \n{y_train_scaled} \ny_train_scaled.shape: {y_train_scaled.shape}")
    # 回归分析
    lr = LinearRegression().fit(x_train_scaled, y_train_scaled)
    y_pred_scaled = lr.predict(x_train_scaled)
    y_pred = scale_y.inverse_transform(y_pred_scaled)
    logger.info(f"y_pred: \n{y_pred} \ny_pred.shape: {y_pred.shape}")
    logger.info(f"coef: {lr.coef_}")
    # 拟合结果画图
    fig = plt.figure(figsize=(8, 8))
    plt.plot(np.array(X_train), np.array(y_train), "o")
    plt.plot(np.array(X_train), np.array(y_pred))
    # plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("线性回归拟合结果")
    plt.tight_layout()
    plt.grid(True)
    plt.show();




# 测试代码 main 函数
def main():
    np.random.seed(42)
    # data
    df = pd.DataFrame({
        "x": np.random.rand(10000),
        "y": np.random.rand(10000),
    })
    logger.info(f"df: \n{df}")
    # model fit
    linear_model(df)

if __name__ == "__main__":
    main()
