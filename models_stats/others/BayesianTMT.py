# -*- coding: utf-8 -*-

# ***************************************************
# * File        : bayesian_tmf.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-01
# * Version     : 0.1.110123
# * Description : 多变量时序数据贝叶斯矩阵分解, 用可用于异常值填充和预测
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# Python Library
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger
import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod  # 两个具有相同列数的矩阵A_ik与矩阵B_jk的对应列向量的克罗内克积排列而成的ij_k
from scipy.stats import wishart  # 用来描述多元正态分布样本的协方差矩阵
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower  # Cholesky分解
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut  # 三角求解ax=b


class BayesianTMF:
    """
    矩阵分解 + VAR多维时序数据填补与预测模型
    """

    def __init__(self, rank = 10, time_lags = None):
        self.rank = rank  # 矩阵分解维度
        self.time_lags = time_lags  # VAR模型时序序列相对下标
        self.inited = False  # 模型是否已预训练

        self.dim_n = None  # 时序数据空间维度
        self.dim_t = None  # 时序数据时间维度
        self.W = None  # 空间因子矩阵 (dim_n, R)
        self.X = None  # 时间因子矩阵 (dim_t, R)
        self.A = None  # Var系数矩阵
        self.tau = None
        self.beta0 = 1

    def Predict(self, input_series, burn_iter = 1000, gibbs_iter = 100, predict_step = 10, option = "factor"):
        """
        BTMF(Bayesian Temporal Matrix Factorization) for predict
        Parameters:
            input_series: 已有时序数据
            burn_iter: 训练步数
            gibbs_iter: 采样数
            predict_step: 预测步数
            option: 参数更新选项
        Return: 
            预测结果
        """
        # 1. 初始化与数据预处理
        self.Init(input_series)
        processed_data, ind, pos_imputation = self.DataProcessing(input_series)

        # 2. 模型初始化
        d = self.time_lags.shape[0]  # Var模型阶数
        W_plus = np.zeros((self.dim_n, self.rank, gibbs_iter))
        A_plus = np.zeros((self.rank * d, self.rank, gibbs_iter))
        tau_plus = np.zeros((self.dim_n, gibbs_iter))
        sigma_plus = np.zeros((self.rank, self.rank, gibbs_iter))
        temp_hat = np.zeros(len(pos_imputation[0]))  # 异常数据个数初始化

        mat_hat_plus = np.zeros((self.dim_n, self.dim_t))  # 训练后结果
        X_plus = np.zeros((self.dim_t + predict_step, self.rank, gibbs_iter))
        mat_new_plus = np.zeros((self.dim_n, predict_step))  # 预测结果

        for it in range(burn_iter + gibbs_iter):
            tau_ind = self.tau[:, None] * ind  # 利用tau更新ind矩阵
            tau_sparse_mat = self.tau[:, None] * processed_data  # 利用tau更新输入数据矩阵

            # 1. 采样空间矩阵因子
            self.SampleFactorW(tau_ind, tau_sparse_mat)

            # 2. 采样VAR模型系数
            A, sigma = self.SampleVarCoefficient()

            # 3. 采样时间矩阵因子
            self.SampleFactorX(tau_ind, tau_sparse_mat, A, inv(sigma))

            # 4. 计算补全后的矩阵
            mat_hat = self.W @ self.X.T

            # 5. 更新tau
            if option == "factor":
                self.tau = self.SamplePrecisionTau(processed_data, mat_hat, ind)
            elif option == "pca":
                tau = self.SamplePrecisionScalarTau(processed_data, mat_hat, ind)
                self.tau = tau * np.ones(self.dim_n)
            temp_hat += mat_hat[pos_imputation]

            if it + 1 > burn_iter:
                W_plus[:, :, it - burn_iter] = self.W
                A_plus[:, :, it - burn_iter] = A
                sigma_plus[:, :, it - burn_iter] = sigma
                tau_plus[:, it - burn_iter] = self.tau
                mat_hat_plus += mat_hat

                # 预测
                X0 = self.Var4cast(A, self.X, sigma, predict_step)
                X_plus[:, :, it - burn_iter] = X0
                mat_new_plus += self.W @ X0[self.dim_t: self.dim_t + predict_step, :].T

        mat_hat = mat_hat_plus / gibbs_iter
        mat_predict_hat = mat_new_plus / gibbs_iter
        return mat_hat, mat_predict_hat
        # mat_hat = np.append(mat_hat, mat_new_hat, axis = 1)
        # return mat_hat, mat_new_hat, W_plus, X_plus, A_plus, sigma_plus, tau_plus

    def Impute(self, input_series, burn_iter = 1000, gibbs_iter = 100, option = "factor"):
        """
        BTMF(Bayesian Temporal Matrix Factorization) for imputation
        """
        # 1. 初始化与数据预处理
        self.Init(input_series)
        processed_data, ind, pos_imputation = self.DataProcessing(input_series)

        # 2. 模型初始化
        d = self.time_lags.shape[0]  # Var时序个数, Var模型阶数
        W_plus = np.zeros((self.dim_n, self.rank))
        X_plus = np.zeros((self.dim_t, self.rank))
        A_plus = np.zeros((self.rank * d, self.rank))
        mat_hat_plus = np.zeros((self.dim_n, self.dim_t))  # 训练后结果
        temp_hat = np.zeros(len(pos_imputation[0]))  # 异常数据个数初始化

        for it in range(burn_iter + gibbs_iter):
            tau_ind = self.tau[:, None] * ind  # 利用tau更新ind矩阵
            tau_sparse_mat = self.tau[:, None] * processed_data  # 利用tau更新输入数据矩阵

            # 1. 采样空间矩阵因子
            self.SampleFactorW(tau_ind, tau_sparse_mat)

            # 2. 采样VAR模型系数
            A, sigma = self.SampleVarCoefficient()

            # 3. 采样时间矩阵因子
            self.SampleFactorX(tau_ind, tau_sparse_mat, A, inv(sigma))

            # 4. 计算补全后的矩阵
            mat_hat = self.W @ self.X.T

            # 5. 更新tau
            if option == "factor":
                self.tau = self.SamplePrecisionTau(processed_data, mat_hat, ind)
            elif option == "pca":
                tau = self.SamplePrecisionScalarTau(processed_data, mat_hat, ind)
                self.tau = tau * np.ones(self.dim_n)
            temp_hat += mat_hat[pos_imputation]

            # 6. gibbs采样模型参数
            if it + 1 > burn_iter:
                W_plus += self.W
                X_plus += self.X
                A_plus += A
                mat_hat_plus += mat_hat

        # 更新模型参数
        self.W = W_plus / gibbs_iter
        self.X = X_plus / gibbs_iter
        self.A = A_plus / gibbs_iter
        mat_hat = mat_hat_plus / gibbs_iter
        return mat_hat

    def Init(self, input_series):
        """
        模型初始化状态检查已经更新
        """
        dim_n, dim_t = input_series.shape
        if not self.inited:
            self.dim_n, self.dim_t = dim_n, dim_t
            self.W = 0.01 * np.random.randn(dim_n, self.rank)
            self.X = 0.01 * np.random.randn(dim_t, self.rank)
            self.tau = np.ones(dim_n)
            self.inited = True
        else:
            assert dim_n == self.dim_n and dim_t == self.dim_t

    @staticmethod
    def DataProcessing(input_series):
        """
        数据预处理
        Parameters:
            input_series: 原始时序数据
        Returns:
            input_series: 异常处理以后的数据
            ind: True, False表示的输入数据
            pos_imputation: 需要填补的元素的i, j下标
        """
        if not np.isnan(input_series).any():  # 如果输入矩阵没有np.nan无效元素
            ind = input_series != 0  # 0表示异常数据
            pos_imputation = np.where(input_series == 0)
            return input_series, ind, pos_imputation
        elif np.isnan(input_series).any():  # 如果以np.nan表示无效元素
            ind = ~np.isnan(input_series)  # nan表示异常数据
            pos_imputation = np.where(np.isnan(input_series))
            input_series[pos_imputation] = 0
            return input_series, ind, pos_imputation
        else:
            raise ValueError("invalid data should be 0 or np.nan")

    def Var4cast(self, A, X, sigma, multi_step):
        """
        Var Model for forecasting
        Parameters: 
            A: VAR模型矩阵
            X: 已有时序数据
            sigma: 协方差矩阵
            multi_step: 预测步数
        Returns: 
            返回带有预测结果的X
        """
        dim, rank = X.shape
        d = self.time_lags.shape[0]  # Var模型的阶数
        X_new = np.append(X, np.zeros((multi_step, self.rank)), axis = 0)  # 预测后的序列
        # 预测
        for t in range(multi_step):
            var = A.T @ X_new[dim + t - self.time_lags, :].reshape(self.rank * d)  # 均值
            X_new[self.dim_t + t, :] = self.MvnrndPre(var, sigma)  # 采样
        return X_new

    def SampleFactorW(self, tau_ind, tau_sparse_mat):
        """
        采样更新空间因子矩阵
        Parameters:
        """
        W_bar = np.mean(self.W, axis = 0)  # 按列求均值
        temp = self.dim_n / (self.dim_n + self.beta0)

        # W共轭分布参数*
        # 这里W0为单位矩阵(R, R), N*Sw = cov_mat(W, W_bar), mu_0 = 0,所以这里是np.outer
        var_W_hyper = inv(np.eye(self.rank) + self.CovMat(self.W, W_bar) + temp * self.beta0 * np.outer(W_bar, W_bar))

        # lambda共轭分布获得方差的超参, lambda0* = R + N
        var_lambda_hyper = wishart.rvs(df = self.dim_n+self.rank, scale = var_W_hyper)
        var_mu_hyper = self.MvnrndPre(temp * W_bar, (self.dim_n + self.beta0) * var_lambda_hyper)
        vargin = 1 if self.dim_n * self.rank * self.rank > 1E8 else 0
        if vargin == 0:
            var1 = self.X.T  # 时间因子矩阵的转置 (dim_t, R)
            var2 = kr_prod(var1, var1)
            var3 = (var2 @ tau_ind.T).reshape([self.rank, self.rank, self.dim_n]) + var_lambda_hyper[:, :, None]
            var4 = var1 @ tau_sparse_mat.T + (var_lambda_hyper @ var_mu_hyper)[:, None]
            for i in range(self.dim_n):
                mu_i = solve(var3[:, :, i], var4[:, i])
                self.W[i, :] = self.MvnrndPre(mu_i, var3[:, :, i])
        elif vargin == 1:
            for i in range(self.dim_n):
                pos0 = np.where(self.X[i, :] != 0)
                Xt = self.X[pos0[0], :]
                var_mu = self.tau[i] * Xt.T @ self.X[i, pos0[0]] + var_lambda_hyper @ var_mu_hyper
                var_lambda = self.tau[i] * Xt.T @ Xt + var_lambda_hyper
                self.W[i, :] = self.MvnrndPre(solve(var_lambda, var_mu), var_lambda)

    def SampleFactorXPartial(self, tau_ind, tau_sparse_mat, W, X, A, lambda_x, back_step):
        """
        sampling T-by-R factor matrix X.
        """
        hd = np.max(self.time_lags)
        h1 = np.min(self.time_lags)
        d = self.time_lags.shape[0]

        A0 = np.dstack([A] * d)
        for k in range(d):
            A0[k * self.rank: (k+1) * self.rank, :, k] = 0

        mat0 = lambda_x @ A.T
        mat1 = np.einsum("kij, jt -> kit",
                         A.reshape([d, self.rank, self.rank]), lambda_x)
        mat2 = np.einsum("kit, kjt -> ij", mat1,
                         A.reshape([d, self.rank, self.rank]))

        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind[:, -back_step:]).reshape([self.rank,
                                                         self.rank, back_step]) + lambda_x[:, :, None]
        var4 = var1 @ tau_sparse_mat[:, -back_step:]
        for t in range(self.dim_t - back_step, self.dim_t):
            Mt = np.zeros((self.rank, self.rank))
            Nt = np.zeros(self.rank)
            Qt = mat0 @ X[t - self.time_lags, :].reshape(self.rank * d)
            index = list(range(0, d))
            if self.dim_t - hd <= t < self.dim_t - h1:
                index = list(np.where(t + self.time_lags < self.dim_t))[0]
            if t < self.dim_t - h1:
                Mt = mat2.copy()
                temp = np.zeros((self.rank * d, len(index)))
                n = 0
                for k in index:
                    temp[:, n] = X[t + self.time_lags[k] -
                                   self.time_lags, :].reshape(self.rank * d)
                    n += 1
                temp0 = X[t + self.time_lags[index], :].T - \
                    np.einum("ijk, ik -> jk", A0[:, :, index], temp)
                Nt = np.einum("kij, jk -> i", mat1[index, :, :], temp0)
            var3[:, :, t + back_step - self.dim_t] = var3[:,
                                                          :, t + back_step - self.dim_t] + Mt
            X[t, :] = self.MvnrndPre(solve(var3[:, :, t + back_step - self.dim_t], var4[:, t + back_step - self.dim_t] + Nt + Qt), var3[:, :, t + back_step - self.dim_t])

    def SampleFactorX(self, tau_ind, tau_sparse_mat, var_A, lambda_x):
        """
        采样更新时空因子矩阵X
        """
        hd = np.max(self.time_lags)
        h1 = np.min(self.time_lags)
        d = self.time_lags.shape[0]

        A0 = np.dstack([var_A] * d)
        for k in range(d):
            A0[k * self.rank: (k+1) * self.rank, :, k] = 0

        mat0 = lambda_x @ var_A.T
        mat1 = np.einsum("kij, jt -> kit", var_A.reshape([d, self.rank, self.rank]), lambda_x)
        mat2 = np.einsum("kit, kjt -> ij", mat1, var_A.reshape([d, self.rank, self.rank]))

        var1 = self.W.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind).reshape([self.rank, self.rank, self.dim_t]) + lambda_x[:, :, None]
        var4 = var1 @ tau_sparse_mat

        for t in range(self.dim_t):
            Mt = np.zeros((self.rank, self.rank))
            Nt = np.zeros(self.rank)
            Qt = mat0 @ self.X[t - self.time_lags, :].reshape(self.rank * d)
            index = list(range(0, d))
            if self.dim_t - hd <= t < self.dim_t - h1:
                index = list(np.where(t + self.time_lags < self.dim_t))[0]
            elif t < hd:
                Qt = np.zeros(self.rank)
                index = list(np.where(t + self.time_lags >= hd))[0]

            if t < self.dim_t - h1:
                Mt = mat2.copy()
                temp = np.zeros((self.rank * d, len(index)))
                n = 0
                for k in index:
                    temp[:, n] = self.X[t + self.time_lags[k] -
                                        self.time_lags, :].reshape(self.rank * d)
                    n += 1
                temp0 = self.X[t + self.time_lags[index], :].T - np.einsum("ijk, ik -> jk", A0[:, :, index], temp)
                Nt = np.einsum("kij, jk -> i", mat1[index, :, :], temp0)

            var3[:, :, t] = var3[:, :, t] + Mt
            if t < hd:
                var3[:, :, t] - var3[:, :, t] - lambda_x + np.eye(self.rank)
            self.X[t, :] = self.MvnrndPre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

    def SampleVarCoefficient(self):
        """
        采样VAR系数矩阵
        """
        d = self.time_lags.shape[0]
        hd = np.max(self.time_lags)

        Z_mat = self.X[hd:self.dim_t, :]
        Q_mat = np.zeros((self.dim_t - hd, self.rank * d))
        # 对于T长的序列, 最大时间步长尺度为hd, 那么可以迭代T-hd步
        # 遍历获取T-hd每步中对应的Rd的x矩阵
        for k in range(d):
            # k = 0, Q_mat[:, R] = X[hd - h1 : T - h1, :]
            # k = 1, Q_mat[:, R:2R] = X[hd - h2, T - h2, :]
            # ...
            # k = d-1, Q_mat[:, (d-1)R: dR] = X[0:T-hd, :]
            Q_mat[:, k * self.rank: (k + 1) * self.rank] = self.X[hd - self.time_lags[k]: self.dim_t - self.time_lags[k], :]

        var_psi0 = np.eye(self.rank * d) + Q_mat.T @ Q_mat
        var_psi = inv(var_psi0)

        var_M = var_psi @ Q_mat.T @ Z_mat
        var_S = np.eye(self.rank) + Z_mat.T @ Z_mat - \
            var_M.T @ var_psi0 @ var_M
        Sigma = invwishart.rvs(df = self.rank + self.dim_t - hd, scale = var_S)
        return self.Mnrnd(var_M, var_psi, Sigma), Sigma

    @staticmethod
    def SamplePrecisionTau(processed_data, mat_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind, axis = 1)
        var_beta = 1e-6 + 0.5 * \
            np.sum(((processed_data - mat_hat) ** 2) * ind, axis = 1)
        return np.random.gamma(var_alpha, 1 / var_beta)

    @staticmethod
    def SamplePrecisionScalarTau(processed_data, mat_hat, ind):
        var_alpha = 1e-6 + 0.5 * np.sum(ind)
        var_beta = 1e-6 + 0.5 * np.sum(((processed_data - mat_hat) ** 2) * ind)
        return np.random.gamma(var_alpha, 1 / var_beta)

    @staticmethod
    def Mnrnd(M, U, V):
        """
        Generate matrix normal distributed random matrix.
        M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
        """
        m, n = M.shape
        X0 = np.random.randn(m, n)
        P = cholesky_lower(U)
        Q = cholesky_lower(V)
        return M + P @ X0 @ Q.T

    @staticmethod
    def MvnrndPre(mu, Lambda):
        """
        给定均值向量和方差, 求多维变量的先验?
        """
        src = normrnd(size = (mu.shape[0],))
        return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False),
                        src, lower = False, check_finite = False, overwrite_b = True) + mu

    @staticmethod
    def CovMat(mat, mat_bar):
        """
        给定矩阵和矩阵方向均值, 求协方差矩阵
        """
        mat_ = mat - mat_bar
        return mat_.T @ mat_

    @staticmethod
    def ComputeMape(var, var_hat):
        """
        计算相对误差
        """
        return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

    @staticmethod
    def ComputeRmse(var, var_hat):
        """
        计算平方根误差
        """
        return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])


def model_test():
    data = [[1, np.nan, 3, 4, 5, ],
            [2, 3, np.nan, 5, 6],
            [np.nan, 4, 5, 6, 7]]
    series_data = np.asarray(data)
    dim1, dim2 = series_data.shape
    tau = np.ones(dim1)
    model = BTMF(rank = 3, time_lags = np.asarray([1, 2]))
    model.Init(series_data)
    p_data, ind, pos_imputation = model.DataProcessing(series_data)
    logger.info("pos_imputation: ", pos_imputation)
    logger.info("ind: ", ind)
    logger.info("p_data: ", p_data)
    tau_ind = tau[:, None] * ind
    tau_sparse_mat = tau[:, None] * p_data  # 利用tau更新输入数据矩阵
    logger.info(tau_ind)
    logger.info(tau_sparse_mat)

    # # test sample coeff W
    # model.SampleFactorW(tau_ind, tau_sparse_mat)
    # logger.info(model.W)
    #
    # # test sample coeff Var
    # A, sigma = model.SampleVarCoefficient()
    # logger.info(model.X, A, sigma)

    # # test smaple coeff X
    # model.SampleFactorX(tau_ind, tau_sparse_mat, A, sigma)

    # test model
    for b_iter, g_iter in zip([100, 500, 1000], [10, 50, 50]):
        logger.info("=======", b_iter, g_iter)
        # data_imputed = model.Train(input_series=series_data, burn_iter = b_iter, gibbs_iter = g_iter)
        data_imputed, data_predicted = model.Predict(series_data, burn_iter = b_iter, gibbs_iter = g_iter, predict_step = 3)
        logger.info("origin data:", series_data)
        logger.info("data imputed:", data_imputed)
        logger.info("data predicted:", data_predicted)
        break


if __name__ == "__main__":
    model_test()
