# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-26
# * Version     : 1.0.102615
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def pressing_ratio(raw_data):
    ratio = (max(raw_data) - min(raw_data)) / ((max(raw_data) + min(raw_data)) / 2)

    return ratio


def press_data(raw_data, ratio):
    cyclic_press_data = []
    new_min_data = max(raw_data) - (ratio * (max(raw_data) + min(raw_data)) / 2)
    press_ratio = (max(raw_data) - new_min_data) / (max(raw_data) - min(raw_data))
    for i in raw_data:
        new_i = max(raw_data) - (press_ratio * (max(raw_data) - i))
        cyclic_press_data.append(new_i)
    
    return cyclic_press_data
 

def press_all_data(df, target_ratios):
    new_df = pd.DataFrame()
    for date in sorted(set(df["time"].dt.date)):
        print(f"date: {date}")
        df_temp = df.loc[df["time"].dt.date == date, :]
        # print(f"df_temp: \n{df_temp}")
        ratio = pressing_ratio(list(df_temp["value"].values))
        print(f"ratio: {ratio}")
        for target_ratio in target_ratios:
            if ratio > target_ratio:
                print("press...")
                df_temp[f"value_{int(target_ratio * 100)}"] = press_data(raw_data=list(df_temp["value"].values), ratio=target_ratio)
                new_ratio = pressing_ratio(list(df_temp[f"value_{int(target_ratio * 100)}"].values))
                print(f"new_ratio: {new_ratio}")
            else:
                print("no press...")
                df_temp[f"value_{int(target_ratio * 100)}"] = df_temp["value"]
                new_ratio = pressing_ratio(list(df_temp[f"value_{int(target_ratio * 100)}"].values))
                print(f"new_ratio: {new_ratio}")
        new_df = pd.concat([new_df, df_temp], axis=0)
    
    return new_df


# 构造一段有规律的周期性时序数据（类似于sin或cos曲线）
# df = pd.DataFrame()
# time_points = np.arange(0, 4.5, 0.1)
# cyclic_data = np.sin(time_points)  # 使用正弦函数来模拟周期性
# cyclic_data = cyclic_data + 1
# print(cyclic_data)
# df["time"] = time_points
# df["value"] = cyclic_data


# load data
df = pd.read_csv("./dataset/sum_load_20241001_20250930.csv")
df["time"] = pd.to_datetime(df["time"])

# press data
# 一天数据查看
new_df = press_all_data(df=df.loc[(df["time"] >= "2025-08-25 00:00:00") & (df["time"] < "2025-08-26 00:00:00"), :], target_ratios=[0.1, 0.2, 0.3])
# 所有数据处理
# new_df = press_all_data(df=df, target_ratios=[0.1, 0.2, 0.3])
with pd.option_context('display.max_rows', 100, 'display.max_columns', None):
    print(f"new_df: \n{new_df}")

# 数据保存
# new_df.to_csv("./dataset/sum_load_20241001_20250930_press.csv", encoding="utf_8_sig", index=False)


# 绘制周期性时序数据
plt.figure(figsize=(25, 8))
plt.plot(new_df["time"], new_df["value"], label='Cyclic Component')
plt.plot(new_df["time"], new_df["value_10"], label='Cyclic press 10% Component', alpha=0.8)
plt.plot(new_df["time"], new_df["value_20"], label='Cyclic press 20% Component', alpha=0.8)
plt.plot(new_df["time"], new_df["value_30"], label='Cyclic press 30% Component', alpha=0.8)
# plt.gca().legend_ = None
plt.legend()
plt.title('')
plt.xlabel('')
plt.ylabel('')
# plt.xticks([])
# plt.yticks([])
# plt.savefig("./dataset/sum_load_20241001_20250930_press.png", dpi=300)
plt.show()





def main():
    pass

if __name__ == "__main__":
    main()
