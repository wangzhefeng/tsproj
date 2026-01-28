# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-01-25
# * Version     : 1.0.012521
# * Description : description
# * Link        : https://zhuanlan.zhihu.com/p/588830785
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def load_data():
    """
    load data
    """
    # load data
    buoy = pd.read_csv(
        "https://raw.githubusercontent.com/vcerqueira/medium-articles/refs/heads/main/data/smart_buoy.csv", 
        skiprows=[1], 
        parse_dates=["time"],
    )
    # buoy["time"] = buoy["time"].dt.replace(tzinfo=None)
    # filter columns
    buoy.drop(columns=["station_id"], inplace=True)
    # setting time as index
    buoy.set_index("time", inplace=True)
    # resampling to hourly data
    buoy = buoy.resample("H").mean()
    # simplifying column names
    buoy.columns = [
        'PeakP', 'PeakD', 'Upcross', 
        'SWH', 'SeaTemp', 'Hmax', 'THmax', 
        'MCurDir', 'MCurSpd' 
    ]
    # sort features
    target_var = "SWH"
    cols = list(buoy.columns)
    cols.remove(target_var)
    colnames = cols + [target_var]
    buoy = buoy[colnames]
    # print data
    print(buoy)

    return buoy, target_var, colnames

def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    https://github.com/vcerqueira/medium-articles/blob/main/src/tde.py
    Time delay embedding
    Time series for supervised learning

    Args:
        series: time series as pd.Series
        n_lags: number of past values to used as explanatory variables
        horizon: how many values to forecast
        return_Xy: whether to return the lags split from future observations

    Return: 
        pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    # lag features
    n_lags_iter = list(range(n_lags, -horizon, -1))
    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    df.columns = [
        f'{name}(t-{j - 1})'
        if j > 0 else f'{name}(t+{np.abs(j) + 1})'
        for j in n_lags_iter
    ]
    df.columns = [re.sub('t-0', 't', x) for x in df.columns]

    # return X, y
    if not return_Xy:
        return df
    else:
        is_future = df.columns.str.contains('\+')
        X = df.iloc[:, ~is_future]
        Y = df.iloc[:, is_future]
        if Y.shape[1] == 1:
            Y = Y.iloc[:, 0]
        return X, Y

def baseline(buoy, target_var):
    """
    baseline model
    """
    # create lagged features
    buoy_ds = []
    for col in buoy:
        col_df = time_delay_embedding(buoy[col], n_lags=24, horizon=12)
        buoy_ds.append(col_df)
    # concatenating all variables
    buoy_df = pd.concat(buoy_ds, axis=1).dropna()
    print(buoy_df)
    # defining target and explanatory variables
    predictor_vars = buoy_df.columns.str.contains("\(t\-")
    target_vars = buoy_df.columns.str.contains(f"{target_var}\(t\+")
    print(predictor_vars)
    print(target_vars)
    X = buoy_df.iloc[:, predictor_vars]
    y = buoy_df.iloc[:, target_vars]
    print(X)
    print(y)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    # fitting a lgbm model without feature engineering
    model_wo_fe = MultiOutputRegressor(LGBMRegressor())
    model_wo_fe.fit(X_train, y_train)

    # getting forecasts for the test set
    preds_wo_fe = model_wo_fe.predict(X_test)
    print(preds_wo_fe)
    print(preds_wo_fe.shape)

    # evaluating the model
    mape_wo_fe = mape(y_test, preds_wo_fe)
    print(f"MAPE without feature engineering: {mape_wo_fe}")
    
    return X, y

def univariate_feature_extract(X, colnames):
    """
    单变量特征提取：
        - 计算各变量的滚动统计。例如，滚动平均可以用来消除虚假的观测
        - 可以总结每个变量最近的过去值。例如，计算滚动平均来总结最近的情况，或者滚动差量来了解最近的分散程度。
    二元特征提取：
        - 计算变量对的滚动统计，以总结它们的相互作用。例如，两个变量之间的滚动协方差。
    """
    SUMMARY_STATS = {
        "mean": np.mean,
        "sdev": np.std,
    }
    univariate_features = {}
    for col in colnames:
        # get lags for that column
        X_col = X.iloc[:, X.columns.str.startswith(col)]
        # for each summary stat
        for feat, func in SUMMARY_STATS.items():
            # compute stat along the rows
            univariate_features[f"{col}_{feat}"] = X_col.apply(func, axis=1)
    # concatenate features into a dataframe
    univariate_features_df = pd.concat(univariate_features, axis=1)
    print(univariate_features_df)

    return univariate_features_df




# 测试代码 main 函数
def main():
    buoy, target_var, colnames = load_data()

    # baseline
    X, y = baseline(buoy, target_var)
    # univariate
    univariate_features_df = univariate_feature_extract(X, colnames)

if __name__ == "__main__":
    main()
