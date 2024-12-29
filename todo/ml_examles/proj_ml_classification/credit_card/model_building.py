# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_building.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-27
# * Version     : 0.1.022715
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import re
import tensorflow as tf



def make_binary_classification_model(train_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation = "relu", input_shape = (train_features.shape[-1], )),
            tf.keras.layers.Dense(256, activation = "relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation = "relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation = "sigmoid"),
        ]
    )
    model.summary()
    return model




# 测试代码 main 函数
def main():
    from data_load import data_load
    from data_generator import data_generator
    from data_preprocessing import data_normalize
    
    features, targets = data_load()

    (train_features, train_targets), \
    (validation_features, validation_targets) = data_generator(features, targets)
    
    train_features, validation_features = data_normalize(train_features, validation_features)

    model = make_binary_classification_model(train_features)


if __name__ == "__main__":
    main()

