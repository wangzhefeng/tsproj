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
import matplotlib.pyplot as plt
from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_sim_data():
    np.random.seed(123)
    data = sim_randomwalk(
        n_series = 3, 
        timesteps = 200, 
        process_noise = 10, 
        measure_noise = 30
    )
    return data


def lowess_smoother(data):
    # 平滑处理
    smoother = LowessSmoother(smooth_fraction = 0.1, iterations = 1)
    smoother.smooth(data)
    # 生成范围区间
    low, up = smoother.get_intervals("prediction_interval")
    _low, _up = smoother.get_intervals("sigma_interval", n_sigma = 2)
    data["low"] = np.hstack([data["low"], _low[:, [-1]]])
    data["up"] = np.hstack([data["up"], _up[:,[-1]]])
    is_anomaly = np.logical_or(
        data["original"][:, -1] > data["up"][:, -1],
        data["original"][:, -1] < data["low"][:, -1]
    ).reshape(-1, 1)
    # 数据可视化
    plt.figure(figsize = (18, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(smoother.smooth_data[i], linewidth = 3, color = "blue")
        plt.plot(smoother.data[i], ".k")
        plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha = 0.3)
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title(f"timeseries {i + 1}")
        plt.show()
    
    return smoother.smooth_data






# 测试代码 main 函数
def main():
    ts_data = get_sim_data()
    print(ts_data)
    smoothed_data = lowess_smoother(ts_data)
    print(smoothed_data)



if __name__ == "__main__":
    main()

