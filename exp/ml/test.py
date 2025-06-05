# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-04
# * Version     : 1.0.060414
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from warnings import simplefilter
simplefilter("ignore")
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger







# 测试代码 main 函数
def main():
    # data path
    data_dir = Path("./dataset/ETT-small/")
    logger.info(f"data_dir: {data_dir}")

    # data
    df = pd.read_csv(
        data_dir / "ETTh1.csv",
        parse_dates=["date"],
        index_col="date"
    )
    logger.info(f"df: \n{df.head()}")
    logger.info(f"df na: \n{df.isna().sum()}")
    OT = df[["OT"]]
    logger.info(f"OT: \n{OT}")
    
    # data split
    tscv = TimeSeriesSplit(n_splits=5)
    tscv.split(OT)


if __name__ == "__main__":
    main()
