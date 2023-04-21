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
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))
if os.path.join(_path, "../..") not in sys.path:
    sys.path.append(os.path.join(_path, "../.."))

import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.datasets import AirPassengersDataset

from data.timeseries_data.AirPassengers import darts_data


# data
series = TimeSeries.from_dataframe(darts_data, "Month", "#Passengers")
series = AirPassengersDataset().load()

# data split
train, val = series[:-36], series[-36:]

# model train
model = ExponentialSmoothing()
model.fit(train)

# model predict
prediction = model.predict(len(val), num_samples = 1000)
print(f"prediction = {prediction}")

# data plot
series.plot()
prediction.plot(label = "forecast", low_quantile = 0.05, high_quantile = 0.95)
plt.legend()
plt.show()




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

