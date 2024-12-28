# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TimeSeriesSplit.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-10
# * Version     : 0.1.091017
# * Description : 时间序列分割
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TimeSeriesSplit:
    pass


class TimeSeriesSplitGap:
    pass


class TimeSeriesSplitSliding:
    pass




# 测试代码 main 函数
def main():
    # ------------------------------
    # 
    # ------------------------------
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    X = np.array([
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
    ])
    y = np.array([1, 2, 3, 4, 5, 6])

    tscv = TimeSeriesSplit(n_splits = 5, max_train_size = None, test_size = None, gap = 0)

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test: index={test_index}")
    # ------------------------------
    # 
    # ------------------------------
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    X = np.array([
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
    ])
    y = np.array([1, 2, 3, 4, 5, 6])

    tscv = TimeSeriesSplit(n_splits = 3, test_size = 2, gap = 2)

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test: index={test_index}")
    # ------------------------------
    # 
    # ------------------------------
    
if __name__ == "__main__":
    main()
