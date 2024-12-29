# -*- coding: utf-8 -*-


# ***************************************************
# * File        : optuna_src.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040801
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import sklearn
from sklearn import datasets
import optuna


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]




def objective(trial):
    regressor_name = trial.suggest_categorical(
        'regressor', 
        ['SVR', 'RandomForest']
    )
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log = True)
        regressor_obj = sklearn.svm.SVR(C = svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth = rf_max_depth, n_estimators = 10)
    
    # data
    X, y = datasets.fetch_california_housing(return_X_y = True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state = 0)

    # model training
    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study(direction = "maximize")  # Create a new study.
study.optimize(objective, n_trials = 100)  # Invoke optimization of the objective function.
print(study.best_params)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
