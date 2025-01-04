# -*- coding: utf-8 -*-


# ***************************************************
# * File        : pyod_knn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-31
# * Version     : 0.1.033117
# * Description : description
# * Link        : https://pyod.readthedocs.io/en/latest/example.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyod.models.knn import KNN
from pyod.utils.data import (
    generate_data,
    evaluate_print,
)
from pyod.utils.example import visualize


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
contamination = 0.1  # percentage of outliers
n_train = 200
n_test = 100
X_train, X_test, y_train, y_test = generate_data(
    n_train = n_train,
    n_test = n_test,
    contamination = contamination,
)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# ------------------------------
# model train and valid
# ------------------------------
# train KNN detector
clf_name = "KNN"
clf = KNN()
clf.fit(X_train)

# 训练数据的 prediction labels 和 outlier score
y_train_pred = clf.labels_  # binary labels (0:inliers, 1:outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# 测试数据的 prediction
y_test_pred = clf.predict(X_test)  # outlier labels (0, 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# 测试数据 prediction confidence
y_test_pred, y_test_pred_confidence = clf.predict(
    X_test, 
    return_confidence = True
)  # outlier labels (0 or 1) and confidence in the range of [0,1]

# ------------------------------
# model evaluate
# ------------------------------
# results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# visualize
visualize(
    clf_name, 
    X_train, y_train, 
    X_test, y_test, 
    y_train_pred, y_test_pred, 
    show_figure = True, 
    save_figure = False
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
