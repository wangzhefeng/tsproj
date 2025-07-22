# -*- coding: utf-8 -*-

# ***************************************************
# * File        : kaggle_ts_linear_regression.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-01
# * Version     : 1.0.060118
# * Description : description
# * Link        : link
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
from pathlib import Path
from warnings import simplefilter
simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.plot_results import scatter_reg_plot, model_result_plot
from utils.log_util import logger


# ------------------------------
# data path
# ------------------------------
data_dir = Path("./dataset/ts_course_data")
logger.info(f"data_dir: {data_dir}")

# ------------------------------
# linear regression: tunnel traffic 
# ------------------------------
# data
tunnel = pd.read_csv(
    data_dir / "tunnel.csv", 
    index_col="Day",
    parse_dates=["Day"]
)
tunnel = tunnel.to_period()
df = tunnel.copy()
logger.info(f"tunnel: \n{tunnel}")


# time-step features: linear model
# ------------------------------
# time-step features
df["Time"] = np.arange(len(tunnel.index))
logger.info(f"df: \n{df}")

# time-step feature plot
scatter_reg_plot(df, "Time", "NumVehicles")

# data split
X = df.loc[:, ["Time"]]
y = df.loc[:, "NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# model
model = LinearRegression()
model.fit(X, y)

# model fit
y_fit = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_fit: \n{y_fit}")

# model results plot
model_result_plot(
    y_train=y,
    y_test=None,
    y_fit=y_fit,
    y_pred=None,
    y_fore=None,
    xlabel="Time",
    ylabel="NumVehicles",
    title="Time Plot of Tunnel Traffic Forecast",
)


# lag features: linear model
# ------------------------------
# lag features
df["Lag_1"] = df["NumVehicles"].shift(1)
df = df.dropna(axis=0)
logger.info(f"df: \n{df}")

# lag feature plot
scatter_reg_plot(df, "Lag_1", "NumVehicles", aspect=True)

# data split
X = df[["Lag_1"]]
y = df["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# model fit
model = LinearRegression()
model.fit(X, y)
y_fit = pd.Series(model.predict(X), index=X.index)
logger.info(f"y_fit: \n{y_fit}")

# model results plot
model_result_plot(
    y_train=y,
    y_test=None,
    y_fit=y_fit,
    y_pred=None,
    y_fore=None,
    xlabel="Lag_1",
    ylabel="NumVehicles",
    title="Lag_1 Plot of Tunnel Traffic Forecast",
)


# time-step and lag features: linear model
# ------------------------------
# data split
X = df[["Time", "Lag_1"]]
y = df["NumVehicles"]
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

scaler = StandardScaler()
scaler_target = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler_target.fit_transform(y.values.reshape(-1, 1))
# logger.info(f"X_scaled: \n{X_scaled}")
# logger.info(f"y_scaled: \n{y_scaled}")

# model fit
model = LinearRegression()
model.fit(X, y)
y_fit = model.predict(X)
y_fit = pd.Series(y_fit, index=y.index)
# y_fit = scaler_target.inverse_transform(model.predict(X_scaled))
# y_fit = pd.Series(y_fit.squeeze(), index=y.index)

# model results plot
model_result_plot(
    y_train=y,
    y_test=None,
    y_fit=y_fit,
    y_pred=None,
    y_fore=None,
    xlabel="Time",
    ylabel="NumVehicles",
    title="Time and Lag_1 Plot of Tunnel Traffic Forecast",
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
