# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mllib_pipeline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040819
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# spark session
spark = SparkSession \
    .builder \
    .master(master = "local[4]") \
    .appName("Logistic Regression") \
    .config() \
    .getOrCreate()

# ------------------------------
# training data
# ------------------------------
training = spark.createDataFrame(
    [
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0),
    ],
    ["id", "text", "label"]
)

# ------------------------------
# pipeline
# ------------------------------
# feature engine
tokenizer = Tokenizer(inputCol = "text", outputCol = "words")
hashingTF = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = "features")
# model
lr = LogisticRegression(maxIter = 10, regParam = 0.001)
# pipeline
pipeline = Pipeline(stages = [
    tokenizer, 
    hashingTF, 
    lr
])
# model fit
model = pipeline.fit(training)

# ------------------------------
# test
# ------------------------------
test = spark.createDataFrame(
    [
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop"),
    ],
    ["id", "text"]
)

prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print(f"({rid}, {text}) --> prob={str(prob)}, prediction={prediction}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
