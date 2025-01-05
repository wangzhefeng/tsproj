# -*- coding: utf-8 -*-


# ***************************************************
# * File        : tods_usage.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-31
# * Version     : 0.1.033114
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    accuracy_score,
)
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from axolotl.backend.simple import SimpleRunner
from tods import (
    generate_dataset, 
    generate_problem,
    load_pipeline,
    evaluate_pipeline,
)
from tods.searcher import BruteForceSearch
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
