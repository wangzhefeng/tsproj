# -*- coding: utf-8 -*-


# ***************************************************
# * File        : KMeans.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-13
# * Version     : 0.1.041315
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from darts.datasets import ETTh2Dataset
from darts.ad import KMeansScorer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]
train, val = series.split_before(split_point = 0.6)

# model
scorer = KMeansScorer(k = 2, window = 5)
scorer.fit(train)
anom_score = scorer.score(val)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
