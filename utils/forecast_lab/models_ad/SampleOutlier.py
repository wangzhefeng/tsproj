# -*- coding: utf-8 -*-


# ***************************************************
# * File        : SampleOutlier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-06
# * Version     : 0.1.040618
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

mpl.rcParams['contour.negative_linestyle'] = 'solid'


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class OutlierPreprocessing(object):
    """
    异常值处理类、方法
    """
    def __init__(self, data):
        self.data = data
        self.outliers_index = []
        self.feature_change = []

    def outlier_detect_box(self, n):
        """
        箱型图异常值检测
        Args:
            n: 
        Example:
            data = pd.read.csv("data.csv")
            outliers_to_drop = outlier_detect_box(2)
            data = data.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
        """
        for col in self.data.columns.tolist():
            Q1 = np.percentile(self.data[col], 25)
            Q3 = np.percentile(self.data[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_list_col = self.data[(self.data[col] < Q1 - outlier_step) | (self.data[col] > Q3 + outlier_step)].index
            self.outliers_index.extend(outlier_list_col)
        self.outliers_index = Counter(self.outliers_index)
        multiple_outliers = [k for k, v in self.outliers_index.items() if v > n]
        print("multiple_outliers: {}".format(multiple_outliers))

    def outlier_detect(self, model, sigma = 3):
        """
        基于预测模型识别异常值
        Args:
            model ([type]): predict y values using model
            sigma (int, optional): [description]. Defaults to 3.
        Params:
            data
        Returns:
            outliers: 异常值索引列表
        """
        # 数据处理
        X = self.data.iloc[:, 0:-1]
        y = self.data.iloc[:, -1]
        # 构建预测模型
        try:
            y_pred = pd.Series(model.predict(X), index = y.index)
        except:
            model.fit(X, y)
            y_pred = pd.Series(model.predict(X), index = y.index)
        # 构造 Z-statistic
        resid = y - y_pred
        mean_resid = resid.mean()
        std_resid = resid.std()
        Z = (resid - mean_resid) / std_resid
        self.outliers_index = Z[abs(Z) > sigma].index
        # 打印结果
        print("{} R2={}".format(model, model.score(X, y)))
        print("{} MSE={}".format(model, mean_squared_error(y, y_pred)))
        print("-" * 100)
        print("mean of residuals: {}".format(mean_resid))
        print("std of residuals: {}".format(std_resid))
        print("-" * 100)
        print("{} outliers:\n{}".format(len(self.outliers_index), self.outliers_index.tolist()))
        return y, y_pred, Z


    def outlier_visual(self, y, y_pred, Z):
        """
        可视化基于预测模型识别的异常值
        """
        plt.figure(figsize = (15, 5))

        ax_131 = plt.subplot(1, 3, 1)
        plt.plot(y, y_pred, ".")
        plt.plot(y.loc[self.outliers_index], y_pred.loc[self.outliers_index], "ro")
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("y")
        plt.ylabel("y_pred");
        ax_132 = plt.subplot(1, 3, 2)
        plt.plot(y, y - y_pred, ".")
        plt.plot(y.loc[self.outliers_index], y.loc[self.outliers_index] - y_pred.loc[self.outliers_index], "ro")
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("y")
        plt.ylabel("y - y_pred");
        ax_133 = plt.subplot(1, 3, 3)
        Z.plot.hist(bins = 50, ax = ax_133)
        Z.loc[self.outliers_index].plot.hist(color = "r", bins = 50, ax = ax_133)
        plt.legend(["Accepted", "Outlier"])
        plt.xlabel("z")
        plt.show()


    def outlier_processing(self, limit_value = 10, method = "box_IQR", percentile_limit_set = 90, changed_feature_box = []):
        """
        异常值处理
        Args:
            limit_value: 最小处理样本个数集合,当独立样本大于 limit_value, 认为是连续特征
            method
            percentile_limit_set
            changed_feature_box
        Params:
            data
        """
        feature_cnt = self.data.shape[1]
        #离群点盖帽
        if method == "box_iqr":
            for i in range(feature_cnt):
                if len(pd.DataFrame(self.data.iloc[:, i]).drop_duplicates()) >= limit_value:
                    q1 = np.percentile(np.array(self.data.iloc[:, i]), 25)
                    q3 = np.percentile(np.array(self.data.iloc[:, i]), 75)
                    top = q3 + 1.5 * (q3 - q1)
                    self.data.iloc[:, i][self.data.iloc[:, i] > top] = top
                    self.feature_change.append(i)
        if method == "self_define":
            if len(changed_feature_box) == 0:
                # 当方法选择为自定义,且没有定义changed_feature_box,则全量数据全部按照percentile_limit_set的分位点大小进行截断
                for i in range(feature_cnt):
                    if len(pd.DataFrame(self.data.iloc[:, i]).drop_duplicates()) >= limit_value:
                        q_limit = np.percentile(np.array(self.data.iloc[:, i]), percentile_limit_set)
                        self.data.iloc[:, i][self.data.iloc[:, i]] = q_limit
                        self.feature_change.append(i)
            else:
                # 如果定义了changed_feature_box, 则将changed_feature_box里面的按照box方法, changed_feature_box的feature index按照percentile_limit_set的分位点大小进行截断
                for i in range(feature_cnt):
                    if len(pd.DataFrame(self.data.iloc[:, 1]).drop_duplicates()) >= limit_value:
                        if i in changed_feature_box:
                            q1 = np.percentile(np.array(self.data.iloc[:, i]), 25)
                            q3 = np.percentile(np.array(self.data.iloc[:, i]), 75)
                            top = q3 + 1.5 * (q3 - q1)
                            self.data.iloc[:, i][self.data.iloc[:, i] > top] = top
                            self.feature_change.append(i)
                        else:
                            q_limit = np.percentile(np.array(self.data.iloc[:, i]), percentile_limit_set)
                            self.data.iloc[:, i][self.data.iloc[:, i]] = q_limit
                            self.feature_change.append(i)

        elliptic_envelope = EllipticEnvelope(contamination = outliers_fraction)
        one_class_svm = OneClassSVM(nu = outliers_fraction, kernel = 'rbf', gamma = 0.1)
        isolation_forest = IsolationForest(behaviour = 'new', contamination = outliers_fraction, random_state = 42)
        local_outlier_factor = LocalOutlierFactor(n_neighbors = 35, contamination = outliers_fraction)






# 测试代码 main 函数
def main():
    from sklearn.linear_model import Ridge

    train_data_file = "/Users/zfwang/machinelearning/mlproj/src/utils/data/zhengqi_train.txt"
    train_data = pd.read_csv(train_data_file, sep = "\t", encoding = "utf-8")

    outlier_preprocessing = OutlierPreprocessing(train_data)
    y, y_pred, Z = outlier_preprocessing.outlier_detect(model = Ridge(), sigma = 3)
    outlier_preprocessing.outlier_visual(y = y, y_pred = y_pred, Z = Z)

if __name__ == "__main__":
    main()
