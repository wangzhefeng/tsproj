# -*- coding: utf-8 -*-


# ***************************************************
# * File        : dataframe.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040820
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyspark.sql.types import (
    # int
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    # float
    FloatType,
    DoubleType,
    DecimalType,
    # string
    StringType,
    BinaryType,
    BooleanType,
    # datetime
    TimestampType,
    DateType,
    DayTimeIntervalType,
    # array, map
    ArrayType,
    MapType,
    # struct
    StructType,
    StructField,
)
from pyspark.ml.linalg import Vectors


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
