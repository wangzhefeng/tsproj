# -*- coding: utf-8 -*-


# ***************************************************
# * File        : pyod_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-31
# * Version     : 0.1.033116
# * Description : description
# * Link        : https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyod.models.suod import SUOD
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
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
contamination = 0.1
n_train = 200
n_test = 100
X_train, X_test, y_train, y_test = generate_data(
    n_train = n_train,
    n_test = n_test,
    n_features = 2,
    contamination = contamination,
    random_state = 42,
)

# ------------------------------
# model training
# ------------------------------
# train SUOD
clf_name = "SUOD"

# 初始化异常检测器
detector_list = [
    LOF(n_neighbors = 15),
    LOF(n_neighbors = 20),
    LOF(n_neighbors = 25),
    LOF(n_neighbors = 35),
    COPOD(),
    IForest(n_estimators = 100),
    IForest(n_estimators = 200),
]

# 模型选择
clf = SUOD(
    base_estimators = detector_list,
    n_jobs = 2,
    combination = "average",
    verbose = False,
)
clf.fit(X_train)

# ------------------------------
# model validation
# ------------------------------
# train prediction
y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

# test prediction
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

# model evaluate
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# visualize result
visualize(
    clf_name, 
    X_train, y_train, 
    X_test, y_test, 
    y_train_pred, y_test_pred,
    show_figure = True,
    save_figure = False,
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
