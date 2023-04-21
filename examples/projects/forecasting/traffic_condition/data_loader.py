# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-27
# * Version     : 0.1.032717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config.config_loader import settings


# TODO 下载数据报错
def data_loader_org():
    url = settings["PATH"]["data_online_dir"]
    data_dir = tf.keras.utils.get_file(origin = url, extract = True, archive_format = "zip")
    data_dir = data_dir.rstrip(".zip")

    route_distances = pd.read_csv(os.path.join(data_dir, "W_228.csv"), header =  None).to_numpy()
    speeds_array = pd.read_csv(os.path.join(data_dir, "V_228.csv"), header = None).to_numpy()
    print(f"route_distances shape={route_distances.shape}")
    print(f"speeds_array shape={speeds_array.shape}")


def _data_loader_small(route_distances, speeds_array):
    sample_routes = [
        0,
        1,
        4,
        7,
        8,
        11,
        15,
        108,
        109,
        114,
        115,
        118,
        120,
        123,
        124,
        126,
        127,
        129,
        130,
        132,
        133,
        136,
        139,
        144,
        147,
        216,
    ]
    route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
    speeds_array = speeds_array[:, sample_routes]
    print(f"small route_distances shape={route_distances.shape}")
    print(f"small speeds_array shape={speeds_array.shape}")

    return route_distances, speeds_array


def data_loader(is_small: bool = True):
    data_dir = os.path.join(
        settings["PATH"]["data_root_dir"], 
        settings["PATH"]["data_local_dir"],
        settings["PATH"]["data_fold"],
    )
    route_distances = pd.read_csv(
        os.path.join(data_dir, "W_228.csv"), 
        header =  None
    ).to_numpy()
    speeds_array = pd.read_csv(
        os.path.join(data_dir, "V_228.csv"), 
        header = None
    ).to_numpy()
    print(f"route_distances shape={route_distances.shape}")
    print(f"speeds_array shape={speeds_array.shape}")
    
    if is_small:
        route_distances, speeds_array = _data_loader_small(
            route_distances, 
            speeds_array
        )

    return route_distances, speeds_array






# 测试代码 main 函数
def main():
    # data_loader_org()
    route_distances, speeds_array = data_loader()

if __name__ == "__main__":
    main()

