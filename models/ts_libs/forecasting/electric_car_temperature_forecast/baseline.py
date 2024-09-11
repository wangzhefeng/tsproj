# -*- coding: utf-8 -*-


# ***************************************************
# * File        : baseline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-16
# * Version     : 0.1.081622
# * Description : description
# * Link        : link
# * Requirement : pip install pandas
# *               pip install pmdarima
# ***************************************************


# python libraries
from logging import root
import os
from random import sample
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if _path not in sys.path:
    sys.path.append(_path)

import numpy as np
import pandas as pd
from tqdm import tqdm
from pmdarima.arima import auto_arima

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
root_dir = "/Users/zfwang/learn/ml/tsproj"
train_path = os.path.join(root_dir, "data/电动汽车永磁同步电机温度预测挑战赛公开数据/train.csv")
test_path = os.path.join(root_dir, "data/电动汽车永磁同步电机温度预测挑战赛公开数据/test.csv")
sample_submit_path = os.path.join(root_dir, "data/电动汽车永磁同步电机温度预测挑战赛公开数据/sample_submit.csv")


def data_read(train_path: str, test_path: str, sample_submit_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submit = pd.read_csv(sample_submit_path)

    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"sample submit shape: {sample_submit.shape}")
    print(f"train info: {train.info()}")

    return train, test, sample_submit


train, test, sample_submit = data_read(
    train_path = train_path, 
    test_path = test_path,
    sample_submit_path = sample_submit_path,
)


for session_id in tqdm(sample_submit["session_id"].unique()):
    train_y = train[train["session_id"] == session_id]["pm"].tolist()[::-1]
    model = auto_arima(
        train_y,
        start_p = 1,
        start_q = 1,
        max_p = 9,
        max_q = 6,
        max_d = 3,
        max_order = None,
        seasonal = False,
        test = "adf",
        trace = False,
        error_action = "ignore",
        suppress_warnings = True,
        stepwise = True,
        information_criterion = "bic",
        njob = -1,
    )
    pred_res = model.predict(12)
    sample_submit.loc[sample_submit["session_id"] == session_id, "pm"] = pred_res

sample_submit.to_csv(os.path.join(root_dir, "result.csv"), index = False)



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

