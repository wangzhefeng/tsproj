# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ToSupervisedDiff.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-18
# * Version     : 0.1.031822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from sklearn import base

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ToSupervisedDiff(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col, groupCol, numDiffs, dropna = False) -> None:
        self.col = col
        self.groupCol = groupCol
        self.numDiffs = numDiffs
        self.dropna = dropna
    
    def fit(self, X, y = None):
        self.X = X
        return self

    def transform(self, X):
        tmp = self.X.copy()
        # 做 i 步差分
        for i in range(1, self.numDiffs + 1):
            # tmp[str(i) + "_Week_Ago_Diff_" + "_" + self.col] = tmp.groupby([self.groupCol])[self.col].diff(i)
            tmp[f"{self.col}_diff_{str(i)}"] = tmp[self.col].diff(i)
        
        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop = True)
        
        return tmp




# 测试代码 main 函数
def main():
    import numpy as np
    import pandas as pd 

    # processer
    model = ToSupervisedDiff(
        col = "load",
        groupCol = "unique_id",
        numDiffs = 3,
        dropna = True,
    )

    # data
    df = pd.DataFrame({
        "ts": pd.date_range(start="2024-11-14 00:00:00", end="2024-11-15 00:46:00", freq="15min"),
        "unique_id": range(100),
        "load": np.random.randn(100),
        "load2": np.random.randn(100),
    })
    # df.set_index("ts", drop=True,inplace=True)
    # print(df)
    # print(type(df["load"]))
    # print(df.groupby("unique_id")["load"].shift(0))
    
    # processing
    df_diffs = model.fit_transform(df)
    with pd.option_context("display.max_columns", None):
        print(df_diffs)

if __name__ == "__main__":
    main()
