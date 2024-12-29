# -*- coding: utf-8 -*-


# ***************************************************
# * File        : correlation.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040816
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
import findspark


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 指定 spark_home，指定 Python 路径
spark_home = "/Users/zfwang/.pyenv/versions/3.7.10/envs/ml/lib/python3.7/site-packages/pyspark"
python_path = "/Users/zfwang/.pyenv/versions/3.7.10/envs/ml/bin/python"
findspark.init(spark_home, python_path)


spark = SparkSession \
    .builder \
    .master("local") \
    .appName("correlation") \
    .config() \
    .getOrCreate()

data = [
    (Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
    (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
    (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
    (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)
]
df = spark.createDataFrame(data, ["features"]).collect()
print(df)

# 
r1 = Correlation.corr(df, "features").head()
print(f"Pearson correlation matrix:\n {str(r1[0])}")

# Spearman
r2 = Correlation.corr(df, "features", "spearman").head()
print(f"Spearman correlation matrix:\n {str(r2[0])}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
