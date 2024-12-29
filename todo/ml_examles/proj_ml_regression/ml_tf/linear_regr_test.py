#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'这是一个文档注释'

__author__ = 'Alvin Wang'


#===========================================================
#
#===========================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from linear_regr import LinearRegression as LR


if __name__ == '__main__':
    X, y = make_regression(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    Y_train = y_train.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    regr = LR(X.shape[1])
    regr.fit(X_train, Y_train, val_data=(X_test, Y_test))
    Y_pred = regr.predict(X_test)
    print("Tensorflow R2: ", r2_score(Y_pred.ravel(), Y_test.ravel()))

    lr = LinearRegression()
    y_pred = lr.fit(X_train, y_train).predict(X_test)
    print("Sklearn R2: ", r2_score(y_pred, y_test))