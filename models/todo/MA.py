# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MA.py
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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from random import random

from statsmodels.tsa.arima_model import ARMA

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
data = [x + random() for x in range(1, 100)]

# model
model = ARMA(data, order = (0, 1))
model_fit = model.fit(disp = False)

# model predict
y_hat = model_fit.predict(len(data), len(data))
print(y_hat)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
