# -*- coding: utf-8 -*-

# ***************************************************
# * File        : var_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-04
# * Version     : 1.0.060409
# * Description : 向量自回归模型(VAR)
# * Link        : https://www.statsmodels.org/stable/vector_ar.html#var
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4.5),
    titleweight="bold",
    titlesize=18,
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
train_scatter_plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Train trues",
)
test_scatter_plot_params = dict(
    color="C2",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
    label="Test trues",
)
fit_line_plot_params = dict(
    color="C0",
    linewidth=2,
    legend=True,
    label="Train preds",
)
pred_line_plot_params = dict(
    color="C1",
    linewidth=2,
    legend=True,
    label="Test preds",
)
fore_line_plot_params = dict(
    color="C3",
    linewidth=2,
    legend=True,
    label="Forecast",
)
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


"""
The `VAR` class assumes that the passed time series are stationary. 
Non-stationary or trending data can often be transformed to be stationary 
by first-differencing or some other method. 

For direct analysis of non-stationary time series, 
a standard stable VAR(p) model is not appropriate.
"""

# ------------------------------
# data
# ------------------------------
# data read
mdata = sm.datasets.macrodata.load_pandas().data
logger.info(f"mdata: \n{mdata} \nmdata.columns: \n{mdata.columns}")
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
# data
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pd.DatetimeIndex(quarterly)
logger.info(f"mdata: \n{mdata} \nmdata.columns: \n{mdata.columns}")
# data
data = np.log(mdata).diff().dropna()
logger.info(f"data: \n{data} \ndata.columns: \n{data.columns}")

# ------------------------------
# model
# ------------------------------
# model
model = VAR(data)

# lag order selection
model.select_order(maxlags=15)
results = model.fit(maxlags=15, ic="aic")
logger.info(f"result: \n{results.summary()}")

"""
# model fit
results = model.fit(maxlags=2, method="ols")
logger.info(f"result: \n{results.summary()}")

# plot input timeseries
results.plot()
plt.show();

# plot timeseries autocorrelation function
results.plot_acorr()
plt.show();
"""

# forecasting
lag_order = results.k_ar
logger.info(f"lag_order: {lag_order}")

y_pred = results.forecast(data.values[-lag_order:], 5)
logger.info(f"y_pred: \n{y_pred}")

results.plot_forecast(10)
plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
