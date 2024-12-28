# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tscv.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102023
# * Description : 时间序列交叉验证
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def timeseries_cv(x_train_month, x_train, y_train, n):
    """
    自定义时间序列交叉验证(月)
    Args:
        x_train_month ([type]): [description]
        x_train ([type]): [description]
        y_train ([type]): [description]
        n ([type]): [description]
    """
    def data2list(lst):
        """
        #TODO
        """
        ret = []
        for i in lst:
            ret += i
        return ret
    
    groups = x_train.groupby(x_train_month).groups
    sorted_groups = [value.tolist() for (key, value) in sorted(groups.items())]
    cv = [(np.array(data2list(sorted_groups[i:i+n])), np.array(sorted_groups[i+n])) for i in range(len(sorted_groups) - n)]
    
    return cv




# 测试代码 main 函数
def main():
    x_train = pd.DataFrame(
        list(range(100)), 
        columns = ["col0"]
    )
    y_train = pd.DataFrame(
        [np.random.randint(0, 2) for i in range(100)], 
        columns = ["y"]
    )
    x_train_month = ['2018-01'] * 20 + \
        ['2018-02'] * 20 + \
        ['2018-03'] * 20 + \
        ['2018-04'] * 20 + \
        ['2018-05'] * 20
    n = 3 # 3个月训练, 1个月验证
    cv = timeseries_cv(x_train_month, x_train, y_train, n)
    print(cv)
    
    # 搭配 GridSearchCV使用
    param_test = {
        "max_depth": list(range(5, 12, 2))
    }
    grid_search = GridSearchCV(
        estimator = XGBClassifier(),
        param_grid = param_test,
        cv = cv
    )

if __name__ == "__main__":
    main()
