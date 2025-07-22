# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hybrid_models.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-02
# * Version     : 1.0.060221
# * Description : Hybrid Forecasting with Residuals
# * Link        : https://www.kaggle.com/code/ryanholbrook/hybrid-models
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import (
    CalendarFourier,
    DeterministicProcess,
)
from xgboost import XGBRegressor
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


# ------------------------------
# data path
# ------------------------------
data_dir = Path("./dataset/ts_course_data")
logger.info(f"data_dir: {data_dir}")

# ------------------------------
# data
# ------------------------------
retail = pd.read_csv(
    data_dir / "us-retail-sales.csv", 
    usecols=["Month", "BuildingMaterials", "FoodAndBeverage"],
    parse_dates=["Month"],
    index_col="Month",
).to_period("D").reindex(columns=["BuildingMaterials", "FoodAndBeverage"])
retail = pd.concat({"Sales": retail}, names=[None, "Industries"], axis=1)
logger.info(f"retail: \n{retail} \nretail.columns: \n{retail.columns}")

# ------------------------------
# trend: linear model
# ------------------------------
# target
y = retail.copy()

# trend features
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=2,
    drop=True,
)
X = dp.in_sample()
logger.info(f"X: \n{X}")
logger.info(f"y: \n{y}")

# data split
idx_train, idx_test = train_test_split(y.index, test_size=12*4, shuffle=False)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# model fit
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
y_fit = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y.columns)

# predict
y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y.columns)

# result
axs = y_train.plot(**train_scatter_plot_params, subplots=True, sharex=True)
axs = y_test.plot(**test_scatter_plot_params, subplots=True, sharex=True, ax=axs)
axs = y_fit.plot(**fit_line_plot_params, subplots=True, sharex=True, ax=axs)
axs = y_pred.plot(**pred_line_plot_params, subplots=True, sharex=True, ax=axs)
for ax in axs:
    # ax.legend()
    ax.set_ylabel("Value")
plt.xlabel("Month")
plt.suptitle("Trend")
plt.show();


# ------------------------------
# trend: xgboost
# ------------------------------
# pivot dataset wide to long
X = retail.stack()
logger.info(f"X: \n{X}")

# grab target series
y = X.pop("Sales")
logger.info(f"y: \n{y}")

# turn row labels into categorical feature columns with a label encoding
X = X.reset_index("Industries")
# logger.info(f"X: \n{X}")

# Label encoding for "Industries" feature
for col in X.select_dtypes(["object", "category"]):
    X[col], _ = X[col].factorize()
# logger.info(f"X: \n{X}")

# label encoding for annual seasnality 
X["Month"] = X.index.month
logger.info(f"X: \n{X}")

# data split
# idx_train, idx_test = train_test_split(X.index, test_size=12*4, shuffle=False)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Create residuals (the collection of detrended series) from the training setCreate residuals()
y_fit = y_fit.stack().squeeze()  # trend from training set
y_pred = y_pred.stack().squeeze()  # trend from test set
y_resid = y_train - y_fit

# model training
xgb = XGBRegressor()
xgb.fit(X_train, y_resid)

# Add the predicted residuals onto the predicted trends
y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred

# results
axs = y_train.unstack(['Industries']).plot(
    **train_scatter_plot_params, 
    subplots=True, sharex=True, title=['BuildingMaterials', 'FoodAndBeverage']
)
axs = y_test.unstack(['Industries']).plot(
    **test_scatter_plot_params, 
    subplots=True, sharex=True, ax=axs
)
axs = y_fit_boosted.unstack(['Industries']).plot(
    **fit_line_plot_params,
    subplots=True, sharex=True, ax=axs,
)
axs = y_pred_boosted.unstack(['Industries']).plot(
    **pred_line_plot_params,
    subplots=True, sharex=True, ax=axs,
)
for ax in axs: 
    ax.legend([])
plt.show();






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
