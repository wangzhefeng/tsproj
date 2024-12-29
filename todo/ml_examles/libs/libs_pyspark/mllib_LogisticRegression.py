# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mllib_pipeline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040818
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# spark session
# ------------------------------
spark = SparkSession \
    .builder \
    .master(master = "local[4]") \
    .appName("logistic regression") \
    .config() \
    .getOrCreate()

# ------------------------------
# trainig data
# ------------------------------
training = spark.createDataFrame(
    [
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))
    ], 
    ["label", "features"]
)

# ------------------------------
# model
# ------------------------------
lr = LogisticRegression(maxIter = 10, regParam = 0.01)
print(f"LogisticRegression parameters:\n {lr.explainParam()}\n")

# model 1 fit
model1 = lr.fit(training)
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

# params
paramMap1 = {
    lr.maxIter: 20
}
paramMap1[lr.maxIter] = 30
paramMap1.update({
    lr.regParam: 0.1, 
    lr.threshold: 0.55,
})

paramMap2 = {
    lr.probabilityCol: "myProbability",
}
paramMapCombined = paramMap1.copy()
paramMapCombined.update(paramMap2)

# model 2 fit
model2 = lr.fit(training, paramMapCombined)
print("Model 2 was fit using parameters: ")
print(model2.extractParamMap())

# ------------------------------
# test data
# ------------------------------
test = spark.createDataFrame(
    [
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5]))
    ],
    ["label", "features"]
)
prediction = model2.transform(test)
result = prediction.select(
    "features", 
    "label", 
    "myProbability", 
    "prediction"
).collect()

for row in result:
    print(f"features={row.features}, \
            label={row.label}, \
            prob={row.myProbability}, \
            prediction={row.prediction}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
