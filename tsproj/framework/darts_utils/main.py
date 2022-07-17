# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-06-29
# * Version     : 0.1.062922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
# import os
# import sys
# _path = os.path.abspath(os.path.dirname(__file__))
# if os.path.join(_path, "..") not in sys.path:
#     sys.path.append(os.path.join(_path, ".."))

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.datasets import AirPassengersDataset


series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]
model = ExponentialSmoothing()
model.fit(train)
prediction  = model.predict(len(val), num_samples = 1000)
print(prediction)

series.plot()
prediction.plot(label = "forecast", low_quantile = 0.05, high_quantile = 0.95)
plt.legend()
plt.show()





# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

