# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mllib_decision_tree.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040821
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# spark session
spark = SparkSession \
    .builder \
    .master(master = "local[4]") \
    .appName("Decision Tree Classifier") \
    .config() \
    .getOrCreate()

# data
data = spark \
    .read \
    .format("libsvm") \
    .load("data/mllib/sample_libsvm_data.txt")

# feature engine
labelIndexer = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol = "features", outputCol = "indexedFeatures", maxCategories = 4).fit(data)

# data split
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# model
dt = DecisionTreeClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures")

# pipeline
pipeline = Pipeline(stages = [
    labelIndexer, 
    featureIndexer, 
    dt
])

# model training
model = pipeline.fit(trainingData)
# model prediction
prediction = model.transform(testData)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
