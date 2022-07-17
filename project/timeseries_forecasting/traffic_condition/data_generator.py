# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_generator.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-27
# * Version     : 0.1.032723
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf

from config.config_loader import settings


"""
# data structure

TS,    value_1,   value_2,  ...,      value_N
t+1,   [v1_{t+1}, v1_{t+1}, v1_{t+1},
t+2,   [v1_{t+2}, v1_{t+2}, v1_{t+2},
...,   [...,      ...,       ...,
t+T,   [v1_{t+T}, v1_{t+T}, v1_{t+T},

t+T+1, pred_{T+1}_{1}, pred_{T+1}_{2}, ..., pred_{T+1}_{N}
t+T+2, pred_{T+2}_{2}, pred_{2}, ..., pred_{2}
...,   ...,      ...,      ..., ...
t+T+h, pred_{T+h}_{h}, pred_{h}, ..., pred_{h}
"""


def create_tf_dataset(data_array: np.ndarray, 
                      input_sequence_length: int,
                      forecast_horizon: int,
                      batch_size: int,
                      shuffle: bool,
                      multi_horizon = True,
                      ) -> tf.data.Dataset:
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """
    # inputs
    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        np.expand_dims(
            data_array[:-forecast_horizon], 
            aixs = -1
        ),
        None,
        sequence_length = input_sequence_length,
        shuffle = False,
        batch_size = batch_size,
    )
    # targets
    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length = target_seq_length,
        shuffle = False,
        batch_size = batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    
    return dataset.prefetch(16).cache()


def data_generator_timeseries(train_array: np.ndarray,
                              validation_array: np.ndarray,
                              test_array: np.ndarray) -> Tuple[tf.data.Dataset]:
    forecast_horizon = settings["DATA"]["forecast_horizon"]
    input_sequence_length = settings["DATA"]["input_sequence_length"]
    batch_size = settings["DATA["]["batch_size"]
    multi_horizon = settings["DATA"]["multi_horizon"]

    train_dataset, validation_dataset = (
        create_tf_dataset(
            data_array, 
            input_sequence_length, 
            forecast_horizon, 
            batch_size = batch_size,
            shuffle = True,
            multi_horizon = multi_horizon,
        )
        for data_array in [train_array, validation_array]
    )
    test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size = test_array.shape[0],
        shuffle = False,
        multi_horizon = multi_horizon,
    )

    return train_dataset, validation_dataset, test_dataset


def compute_adjacency_matrix(route_distances: np.ndarray, 
                             sigma2: float, 
                             epsilon: float) -> bool:
    """
    Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )

    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask







# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

