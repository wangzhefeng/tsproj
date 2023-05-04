# -*- coding: utf-8 -*-


# ***************************************************
# * File        : forescast_baseline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-13
# * Version     : 0.1.041315
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
    datetime_attribute_timeseries,
)
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
)
from darts.metrics import mape, smape


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# for reproducibility
torch.manual_seed(1)
np.random.seed(1)
pd.set_option("display.max_columns", None, "display.max_rows", None)


# -----------------------------
# read data
# -----------------------------
series_air = AirPassengersDataset().load()
series_milk = MonthlyMilkDataset().load()

series_air.plot(label = "Number of air passengers")
plt.show()










# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
