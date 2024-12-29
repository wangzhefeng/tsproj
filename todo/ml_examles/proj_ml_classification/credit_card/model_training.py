# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_training.py
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
import tensorflow as tf
from config.config_loader import settings


def model_training(model, 
                   train_features, 
                   train_targets, 
                   validation_features, 
                   validation_targets, 
                   weight_for_0, 
                   weight_for_1):
    metrics = [
        tf.keras.metrics.FalseNegatives(name = "fn"),
        tf.keras.metrics.FaslePositives(name = "fp"),
        tf.keras.metrics.TrueNegatives(name = "tn"),
        tf.keras.metrics.TruePositives(name = "tp"),
        tf.keras.metrics.Precision(name = "precision"),
        tf.keras.metrics.Recall(name = "recall"),
    ]
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-2),
        loss = "binary_crossentropy",
        metrics = metrics,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.joins(settings["PATH"]["model_path"], "fraud_model_at_epoch_{epoch}.h5")
        )
    ]
    class_weight = {
        0: weight_for_0,
        1: weight_for_1,
    }
    model.fit(
        train_features,
        train_targets,
        batch_size = settings["MODEL"]["batch_size"],
        epochs = settings["MODEL"]["epochs"],
        verbose = 2,
        callbacks = callbacks,
        validation_data = (validation_features, validation_targets),
        clas_weight = class_weight,
    )
    return model




# 测试代码 main 函数
def main():
    from data_load import data_load
    from data_generator import data_generator, weight_generate
    from data_preprocessing import data_normalize
    from model_building import make_binary_classification_model
    
    features, targets = data_load()

    (train_features, train_targets), \
    (validation_features, validation_targets) = data_generator(features, targets)

    weight_for_0, weight_for_1 = weight_generate(train_targets)    

    train_features, validation_features = data_normalize(train_features, validation_features)

    model = make_binary_classification_model(train_features)

    model = model_training(
        model, 
        train_features, 
        train_targets, 
        validation_features, 
        validation_targets, 
        weight_for_0, 
        weight_for_1,
    )    


if __name__ == "__main__":
    main()

