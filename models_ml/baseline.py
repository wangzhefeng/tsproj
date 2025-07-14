# -*- coding: utf-8 -*-

# ***************************************************
# * File        : baseline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102022
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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 数据读取
series = pd.read_csv(
    "E:/projects/timeseries_forecasting/tsproj/dataset/shampoo-sales.csv",
    header = 0,
    index_col = 0,
    parse_dates = [0],
    date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
)
print(series.head())


# 构建一个监督学习结构，滞后数据表
df = pd.concat([series.shift(1), series], axis = 1)
df.columns = ['t-1', 't+1']
print(df.head())


# 将数据分割为训练集和测试集
X = df.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]
print(f"test_X: {test_X}")
print(f"test_y: {test_y}")


# 持久性模型，这里的映射模型是简单的对应关系
def model_persistence(x):
	return x


# 用 model_persistence 函数模拟预测模型，对预测结果和真实值做残差比较，检验结果
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
print(f"Prediction: {predictions}")

test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)


# 画出结果
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
