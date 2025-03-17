# -*- coding: utf-8 -*-


# ***************************************************
# * File        : pyod_fusion.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-31
# * Version     : 0.1.033117
# * Description : description
# * Link        : https://github.com/yzhao062/pyod/blob/master/examples/comb_example.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.knn import KNN
from pyod.models.combination import average, maximization, median, aom, moa
from pyod.utils.data import (
    generate_data,
    evaluate_print,
)
from pyod.utils.utility import standardizer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data
mat_file = "cardio.mat"
try:
    mat = loadmat(os.path.join("./data/", mat_file))
except TypeError:
    print(f"{mat_file} does not exist. Use generated data")
    X, y = generate_data(train_only = True)
except IOError:
    print(f"{mat_file} does not exist. Use generated data")
    X, y = generate_data(train_only = True)
else:
    X = mat["X"]
    y = mat["y"].ravel()

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

# data normalize
X_train_norm, X_test_norm = standardizer(X_train, X_test)

# ------------------------------
# model
# ------------------------------
# init 20 base detectors
k_list = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
]
n_clf = len(k_list)
print(f"Combining {n_clf} KNN detectors.")

# model training
train_scores = np.zeros([X_train.shape[0], n_clf])
test_scores = np.zeros([X_test.shape[0], n_clf])
for i in range(n_clf):
    clf = KNN(n_neighbors = k_list[i], method = "largest")
    clf.fit(X_train_norm)
    # train predict
    train_scores[:, i] = clf.decision_scores_
    test_scores[:, i] = clf.decision_function(X_test_norm)
print(f"Train scores:\n {train_scores}") 
print(f"Test scores:\n {test_scores}")

# model fusion
train_scores_norm, test_scores_norm = standardizer(
    train_scores, 
    test_scores
)

# Average
y_by_average = average(test_scores_norm)
evaluate_print("Combination by Average", y_test, y_by_average)

# Max
y_by_maximization = maximization(test_scores_norm)
evaluate_print("Combination by Maximization", y_test, y_by_maximization)

# Median
y_by_median = median(test_scores_norm)
evaluate_print("Combination by Median", y_test, y_by_median)

# AMO
y_by_amo = aom(test_scores_norm, n_buckets = 5)
evaluate_print("Combination by AOM", y_test, y_by_amo)

# moa
y_by_moa = moa(test_scores_norm, n_buckets = 5)
evaluate_print("Combination by MOA", y_test, y_by_moa)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
