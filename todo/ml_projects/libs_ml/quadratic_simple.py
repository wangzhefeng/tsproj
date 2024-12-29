# -*- coding: utf-8 -*-


# ***************************************************
# * File        : quadratic_simple.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040814
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import optuna


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])

    return x ** 2 + y


def optim():
    print("Running 10 trials...")
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 10)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")

    print("Running 20 additional trials...")
    study.optimize(objective, n_trials = 20)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")

    print("Running additional trials in 2 seconds...")
    study.optimize(objective, timeout = 2.0)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")




# 测试代码 main 函数
def main():
    optim()

if __name__ == "__main__":
    main()
