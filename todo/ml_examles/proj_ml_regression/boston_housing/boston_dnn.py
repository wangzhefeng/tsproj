# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras import models, layers, preprocessing
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical


# 读取数据
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


# 数据预处理[标准化]
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

test_data -= mean
test_data /= std


# build model
def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation = "relu", input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = "relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer = "rmsprop", loss = "mse", metrics = ["mae"])

    return model


def model_training(k, num_epochs):
    """K-fold CV

    Arguments:
        k {[type]} -- [description]
        num_epochs {[type]} -- [description]
    """
    num_val_samples = len(train_data) // k
    all_scores = []
    for i in range(k):
        print("processing fold #", i)
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_labels = train_labels[i * num_val_samples:(i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis = 0)
        partial_train_labels = np.concatenate([train_labels[:i * num_val_samples], train_labels[(i + 1) * num_val_samples:]], axis = 0)
        model = build_model(train_data)
        if num_epochs == 100:
            model.fit(partial_train_data, partial_train_labels, epochs = num_epochs, batch_size = 1, verbose = 0)
            val_mse, val_mae = model.evaluate(val_data, val_labels, verbose = 0)
        elif num_epochs == 500:
            history = model.fit(partial_train_data, partial_train_labels, 
                                validation_data = (val_data, val_labels), 
                                epochs = num_epochs, 
                                batch_size = 1, 
                                verbose = 0)
            val_mae = history.history["val_mae"]
        all_scores.append(val_mae)

    return all_scores


# ------------------
# trainging 1
# ------------------
all_scores = model_training(k = 5, num_epochs = 100)
finall_score = np.mean(all_scores)
print(all_scores)
print(finall_score)


# ------------------
# training 2
# ------------------
all_mae_histories = model_training(k = 5, num_epochs = 500)

average_mae_history = [np.mean(x[i] for x in all_mae_histories) for i in range(500)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs:")
plt.ylabel("Validation MAE")
plt.show()
# 对图形进行调整
def smooth_curve(points, factor = 0.9):
    smooth_points = []
    for point in points:
        previous = smoothed_points[-1]
        smoothed_points.append(previous * factor + point * (1 - factor))
    else:
        smoothed_points.append(point)
    
    return smooth_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


# ------------------
# evaluate on testing data
# ------------------
def evaluate_test_data():
    """使用最佳参数在所有训练数据上训练最终的生产模型，然后观察模型在测试集上的性能"""
    model = build_model(train_data)
    model.fit(train_data, train_labels, epochs = 80, batch_size = 16, verbose = 0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
    print(test_mae_score)



evaluate_test_data()
