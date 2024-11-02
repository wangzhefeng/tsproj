# -*- coding: utf-8 -*-


# ***************************************************
# * File        : timeseries_tsmoothie.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-09
# * Version     : 0.1.020923
# * Description : description
# * Install     : pip install tsmoothie
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_sim_data():
    np.random.seed(123)
    data = sim_randomwalk(
        n_series = 1, 
        timesteps = 200, 
        process_noise = 10, 
        measure_noise = 30
    )
    df = pd.Series(
        data[0],
        pd.date_range(start = "2022-01-01 01:00:00", periods = 200, freq = "H"),
    name = ["value", "dt"])
    return df


def lowess_smoother(data):
    # 平滑处理
    smoother = LowessSmoother(smooth_fraction = 0.1, iterations = 1)
    smoother.smooth(data.values)
    # 生成范围区间
    pred_low, pred_up = smoother.get_intervals("prediction_interval")
    sigma_low, sigma_up = smoother.get_intervals("sigma_interval", n_sigma = 2)
    # print(smoother.smooth_data)
    # print(smoother.data)
    # print(pred_low)
    # print(pred_up)
    data["smooth_data"] = smoother.smooth_data[0].T
    data["data"] = smoother.data[0].T
    data["pred_low"] = pred_low[0].T
    data["pred_up"] = pred_up[0].T
    data["sigma_low"] = sigma_low[0].T
    data["sigma_up"] = sigma_up[0].T
    # is_anomaly = np.logical_or(
    #     data["original"][:, -1] > data["up"][:, -1],
    #     data["original"][:, -1] < data["low"][:, -1]
    # ).reshape(-1, 1)
    
    # 数据可视化
    # fig = plt.figure(figsize = (18, 5))
    # ax = plt.subplot(1, 1, 1)
    # plt.plot(smoother.smooth_data[0], linewidth = 3, color = "blue")
    # plt.plot(smoother.data[0], ".k")
    # plt.fill_between(
    #     range(len(smoother.data[0])), 
    #     pred_low[0], 
    #     pred_up[0], 
    #     alpha = 0.3
    # )
    # plt.xlabel("time")
    # plt.ylabel("value")
    # plt.title(f"timeseries {0 + 1}")
    # plt.show()
    
    return data






# 测试代码 main 函数
def main():
    ts_data = get_sim_data()
    print(ts_data)
    print(ts_data.index)
    print(ts_data.values)
    ts_data.plot()
    plt.show()
    # smoothed_data = lowess_smoother(ts_data)
    # print(smoothed_data)
    # print(smoothed_data)


if __name__ == "__main__":
    main()

