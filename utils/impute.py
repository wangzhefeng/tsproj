# -*- coding: utf-8 -*-


# ***************************************************
# * File        : impute.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 时序数据填补方法
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def SimpleImputer(series: pd.DataFrame) -> pd.DataFrame:
    """
    前后值填补算法

    Args:
        series (pd.DataFrame): 待填充数据

    Returns:
        pd.DataFrame: 缺失值填充后的数据
    """
    series = series.fillna(method = "ffill")
    series = series.fillna(method = "bfill")

    return series


def KnnImputer(series, neighbors = 10):
    """
    基于 KNN 的时序数据填补方法
    """
    knn = KNNImputer(n_neighbors = neighbors)
    res = np.asarray(series).reshape(-1, 1)
    res = knn.fit_transform(res)

    return res.reshape(-1).tolist()


def MatrixImputer(series, features = 5, alpha = 0.1, rho = 0.001, error_threshold = 1e-2, maxIter = 1000):
    """
    基于矩阵填补的时序数据填补方法

    Args:
        series: 时序数据, np.array, 需填补的数据为 np.nan
        features: 特征数
        alpha: 超参
        rho: 超参
        error_threshold: 误差收敛阈值
        maxIter: 最大循环次数
    
    Returns:
        返回填补后的时序数据
    https://zhuanlan.zhihu.com/p/93400890
    """
    impute_data = series.copy()
    where_is_nan = np.isnan(series)
    impute_data[where_is_nan] = 0.0
    impute_num = sum(where_is_nan)

    sparse_mat = impute_data.reshape(-1, features).T
    pos_train = np.where(sparse_mat != 0)
    X = sparse_mat.copy()
    T = np.zeros(sparse_mat.shape)
    last_test_mat = None
    for it in range(maxIter):
        u, s, v = np.linalg.svd(X + T / rho, full_matrices=False)
        vec = s - alpha / rho
        vec[np.where(vec < 0)] = 0
        Z = np.matmul(np.matmul(u, np.diag(vec)), v)
        X = Z - T / rho
        X[pos_train] = sparse_mat[pos_train]
        if last_test_mat is None:
            last_test_mat = X
        else:
            err = X - last_test_mat
            err2 = np.multiply(err, err)
            err_i = np.sum(np.sum(err2))
            if err_i / impute_num < error_threshold:
                break
            else:
                last_test_mat = X
        T = T - rho * (Z - X)
    
    return X.T.reshape(-1, 1)


def SimpleImputer(series_df):
    """
    前后值填补算法
    Parameters: 
        series_df: 待填补数据, pandas dataframe
    Returns:
        返回填补后df
    """
    series_df = series_df.fillna(method="ffill")
    return series_df.fillna(method="bfill")


# def KnnImputer(series_data, neighbors=10):
#     """
#     基于KNN的时序数据填补方法
#     """
#     model = KNNImputer(n_neighbors=neighbors)
#     res = np.asarray(series_data).reshape(-1, 1)
#     res = model.fit_transform(res)
#     return res.reshape(-1).tolist()


def MatrixImputer(series_data, features=5, alpha=0.1, rho=0.001, error_threshold=1e-2, maxIter=1000):
    """
    基于矩阵填补的时序数据填补方法
    Parameters:
        series_data: 时序数据, np.array, 需填补的数据为np.nan
        features: 特征数
        alpha: 超参
        rho: 超参
        error_threshold: 误差收敛阈值
        maxIter: 最大循环次数
    Returns:
        返回填补后的时序数据
    https://zhuanlan.zhihu.com/p/93400890
    """
    impute_data = series_data.copy()
    where_is_nan = np.isnan(series_data)
    impute_data[where_is_nan] = 0.0
    impute_num = sum(where_is_nan)

    sparse_mat = impute_data.reshape(-1, features).T
    pos_train = np.where(sparse_mat != 0)
    X = sparse_mat.copy()
    T = np.zeros(sparse_mat.shape)
    last_test_mat = None
    for it in range(maxIter):
        u, s, v = np.linalg.svd(X + T / rho, full_matrices=False)
        vec = s - alpha / rho
        vec[np.where(vec < 0)] = 0
        Z = np.matmul(np.matmul(u, np.diag(vec)), v)
        X = Z - T / rho
        X[pos_train] = sparse_mat[pos_train]
        if last_test_mat is None:
            last_test_mat = X
        else:
            err = X - last_test_mat
            err2 = np.multiply(err, err)
            err_i = np.sum(np.sum(err2))
            if err_i / impute_num < error_threshold:
                break
            else:
                last_test_mat = X
        T = T - rho * (Z - X)
    return X.T.reshape(-1, 1)





ImputeModels = {
    "SimpleImputer": SimpleImputer,
    "KnnImputer": KnnImputer,
    "MatrixImputer": MatrixImputer
}



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

