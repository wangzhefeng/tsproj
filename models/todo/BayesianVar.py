# -*- coding: utf-8 -*-


# ***************************************************
# * File        : bayesian_var.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-30
# * Version     : 0.1.033021
# * Description : 多变量贝叶斯向量自回归模型(Bayesian Vector Auto Regression), 可用于多变量预测
# * Link        : https://www.sciencedirect.com/science/article/abs/pii/B9780444627315000154
# *             : https://www.pymc-labs.io/blog-posts/bayesian-vector-autoregression/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
from scipy.stats import invwishart
from numpy.linalg import inv
from numpy.random import multivariate_normal


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class BayesianVar(object):
    """
    基于 MNIW 先验的贝叶斯的 VAR 模型
    """
    def __init__(self, var_num = 1, time_lags = (0, 1, 2)):
        self.time_lags = time_lags  # [h1, h2, h3, ..., hd]
        self.d = len(time_lags)  # VAR(d)
        self.R = var_num  # variables number
        self.A = np.random.randn(self.R * self.d, self.R)  # 模型系数矩阵
        self.Sigma = None  # 模型方差

    def Predict(self, X: np.array, predict_steps: int, gibbs_iter:int = 100) -> np.array:
        """
        模型预测

        Args:
            X (np.array): 已知数据, shape=(T, R), T时间步, R个变量
            predict_steps (int): 预测时间步数
            gibbs_iter (int, optional): gibbs 采样数. Defaults to 100.

        Returns:
            np.array: 预测数据, shape=(predict_steps, R), R 个变量
        """ 
        hd = np.max(self.time_lags)
        X_hat = np.zeros((predict_steps, gibbs_iter, self.R))  # gibbs 采样
        for it in range(gibbs_iter):
            X_new = np.zeros((hd + predict_steps, self.R))
            X_new[:hd, :] = X[-hd:, :]
            for t in range(predict_steps):
                X_new[hd + t, :] = multivariate_normal(
                    self.A.T @ X_new[hd + t - self.time_lags, :].reshape(self.R * self.d), 
                    self.Sigma
                )  # 采样轨迹
            X_hat[:, it, :] = X_new[-predict_steps:, :]
        return np.mean(X_hat, axis = 1)

    def Train(self, X: np.array, burn_iter: int = 1000):
        """
        模型训练

        Args:
            X (np.array): 训练数据, shape=(T, R),  其中：T 时间步, R 个变量
            burn_iter (int, optional): _description_. Defaults to 1000.
        """
        # check timeseries length and feature
        T, x_R = X.shape
        assert x_R == self.R

        hd = np.max(self.time_lags)
        Z_mat = X[hd:T, :]
        Q_mat = np.zeros((T - hd, self.R * self.d))
        for k in range(self.d):
            """
            k = 0, Q_mat[:, R] = X[hd - h1:T - h1, :]
            k = 1, Q_mat[:, R:2R] = X[hd - h2:T - h2, :]
            ...
            k = d-1, Q_mat[:, (d-1)R: dR] = X[0:T-hd, :]
            """
            Q_mat[:, (k * self.R):((k + 1) * self.R)] = X[(hd - self.time_lags[k]):(T - self.time_lags[k]), :]

        for it in range(burn_iter):
            # 假定分布情况下, 基于时序数据的结果, 获取 VAR 参数
            var_Psi0 = np.eye(self.R * self.d) + Q_mat.T @ Q_mat  # fai_0 单位阵, (Rd, Rd)
            var_Psi = inv(var_Psi0)
            var_M = var_Psi @ Q_mat.T @ Z_mat  # M0 = 0
            var_S = np.eye(self.R) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M  # S0 = 单位阵
            self.Sigma = invwishart.rvs(df = self.R + T - hd, scale = var_S)
            self.A = self.Mnrnd(var_M, var_Psi, self.Sigma)

    def Load(self, model_path):
        """
        模型加载
        """
        data = np.load(model_path)
        self.time_lags = data['time_lags']
        self.d = data['d']
        self.R = data['R']
        self.A = data['A']
        self.Sigma = data['sigma']

    def Save(self, out_path):
        """
        模型保存
        """
        np.savez(
            out_path, 
            time_lags = self.time_lags, 
            d = self.d, 
            R = self.R, 
            A = self.A, 
            sigma = self.Sigma
        )

    @staticmethod
    def Mnrnd(M, U, V):
        """
        Generate normal distributed random matrix
            M is a m-by-n matrix
            U is a m-by-m matrix
            V is a n-by-n matrix
        """
        m, n = M.shape
        X0 = np.random.rand(m, n)
        P = np.linalg.cholesky(U)
        Q = np.linalg.cholesky(V)
        return M + P @ X0 @ Q.T




# 测试代码 main 函数
def main():
    import matplotlib.pyplot as plt

    # data
    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    X1 = np.sin(x) + np.random.rand(100, 1) * 0.5
    X2 = np.sin(1.5 * x) + np.random.rand(100, 1) * 0.2
    X3 = 1.5 * X1
    X = np.hstack([X1, X2, X3])
    print(X.shape)

    # model
    model = BayesianVar(var_num = X.shape[1], time_lags = np.asarray([1, 5, 10, 20, 50]))
    # model training 
    model.Train(X, burn_iter = 1000)
    # model predict
    X_predict = model.Predict(X, predict_steps = 20, gibbs_iter = 100)
    print(X_predict.shape)

    # result plot
    X_all = np.vstack([X, X_predict])
    print(X_all.shape)
    for i in range(3):
        x_i = X_all[:, i].tolist()
        plt.plot(x_i)
    plt.show()

    # 模型保存
    model.Save("./models/var.npz")

    # 模型加载 
    model = BayesianVar()
    model.Load("./models/var.npz")

if __name__ == "__main__":
    main()
