# -*- coding: utf-8 -*-

# ***************************************************
# * File        : baseline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-17
# * Version     : 0.1.051721
# * Description : 1.将单变量时间序列数据转换为监督学习问题
# *               2.建立训练集和测试集
# *               3.定义持久化模型
# *               4.进行预测并建立 baseline 性能
# *               5.查看完整的示例并绘制输出
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import pickle
import random
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
# import paddle
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import tqdm
from loguru import logger
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["SimHei"]
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# 设置随机数种子
fix_seed = 42
# random/np.random
random.seed(fix_seed)
np.random.seed(fix_seed)
# torch
torch.manual_seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fix_seed)
# paddle
# paddle.seed(fix_seed)


# ------------------------------
# data
# ------------------------------
# data load
series = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
    header = 0,
    parse_dates = [0],
    index_col = 0,
    squeeze = True,
    date_parser = lambda dates: datetime.strptime("190" + dates, "%Y-%m")
)
logger.info(series)
series.plot()
# plt.show()

# create lagged dataset
values = pd.DataFrame(series.values)
df = pd.concat([values.shift(1), values], axis = 1)
df.columns = ["t-1", "t+1"]
logger.info(df)

# split into train and test sets
X = df.values
train_size = int(len(X) * 0.7)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]
logger.info(X)
logger.info(train_X)
logger.info(train_y)
logger.info(test_X)
logger.info(test_y)

# ------------------------------
# model
# ------------------------------
# persistence model
def model_persistence(x):
    return x

# ------------------------------
# model validate
# ------------------------------
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
logger.info("Test MSE: %.3f" % test_score)

# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
# plt.show()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
