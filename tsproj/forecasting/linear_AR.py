# -*- coding: utf-8 -*-


# ***************************************************
# * File        : linear_AR.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 基于线性自回归的单变量时序数据预测方法
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
from sklearn.linear_model import LinearRegression


class LinearVAR:
    """
    线性VAR模型用于时序数据的预测, 目标变量x, 可添加多个特征变量, 每个特征变量的开始位置与目标位置一致
    目标变量特征: x_hd, ..., x_h2, x_h1 [h1, h2, ..., hd]
    特征变量: y_gd, ..., y_g2, y_g1 [g1, g2, ..., gd]
    ...
    """
    def __init__(self, predict_n=15):
        self.predict_n = predict_n  # 时序预测的长度
        self.fea_x = []
        self.pred_x = []
        self.idx_list = []
        self.fea_list = []
        self.model = None

    def Predict(self, fea_array):
        """
        模型的预测
        Parameters:
            fea_array: [x_h1, x_h2, ...x_hd, y_yh1, y_h2, ....]
        Returns:
            predict series, list
        """
        return self.model.predict(fea_array)[0]

    def Fit(self):
        """
        模型训练
        """
        self.model = LinearRegression()
        x_fea = np.asarray(self.fea_x)
        if self.fea_list:
            other_fea = np.hstack(self.fea_list)
            train_fea = np.hstack((x_fea, other_fea))
        else:
            train_fea = x_fea
        target_fea = np.asarray(self.pred_x)
        self.model.fit(train_fea, target_fea)

    def AddTargetFea(self, x_data, time_lags=(0, 1, 2)):
        """
        用x_data中t-hd,...t-h1, t的结果, 预测t+1、t+2, t+n的结果(相当于自回归)
        Parameters:
            x_data: list T个时间步, 1个变量
            time_lags = [h1, h2, h3, h4, ..., hd]
        Returns: 
            x_train: T - hd - predict_n, d * h
        """
        hd, R = max(time_lags), len(time_lags)
        T = len(x_data)
        assert hd <= T - self.predict_n
        self.idx_list = list(range(hd, T - self.predict_n))
        for idx in self.idx_list:
            xi = [x_data[idx - i] for i in time_lags]
            self.fea_x.append(xi)
            target_i = x_data[idx+1:idx+self.predict_n+1]
            self.pred_x.append(target_i)

    def AddFeature(self, y_data, time_lags=(0, 1, 2)):
        """
        用y_data中t-hd,...t-h1, t的结果, 作为特征, 用于x的预测
        y_data: list T个时间步, 1个变量
        time_lags = [h1, h2, h3, h4, ..., hd]
        """
        y_hd = max(time_lags)
        assert y_hd <= self.idx_list[0]
        fea_y = []
        for idx in self.idx_list:
            fea_i = [y_data[idx - x] for x in time_lags]
            fea_y.append(fea_i)
        self.fea_list.append(fea_y)


def main():
    x = list(range(10))
    y = [x + 5 for x in range(10)]
    var_model = LinearVAR(predict_n=3)
    var_model.AddTargetFea(x)
    print(var_model.fea_x, var_model.pred_x)
    var_model.AddFeature(y, time_lags=[0])
    print(var_model.fea_list)
    var_model.Fit()

    pred_x = np.asarray([[4, 2, 1, 5]])
    pred_y = var_model.Predict(pred_x)
    print(type(pred_y), pred_y)




if __name__ == "__main__":
    main()




