# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test4.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-26
# * Version     : 1.0.102619
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


import numpy as np
import matplotlib.pyplot as plt

# 构造一段有规律的周期性时序数据（类似于sin或cos曲线）
time_points = np.arange(0, 4.5, 0.1)
cyclic_data = np.sin(time_points)  # 使用正弦函数来模拟周期性
cyclic_data = cyclic_data + 1
cyclic_press_data = []
new_min_data = max(cyclic_data) - (1.0 * (max(cyclic_data) + min(cyclic_data))/2)
press_ratio = (max(cyclic_data) - new_min_data) / (max(cyclic_data) - min(cyclic_data))
print(press_ratio)
for i in cyclic_data:
    new_i = max(cyclic_data) - (press_ratio * (max(cyclic_data) - i))
    cyclic_press_data.append(new_i)

# 绘制周期性时序数据
plt.figure(figsize=(10, 5))
plt.plot(time_points, cyclic_data, label='Cyclic Component', color='orange')
plt.plot(time_points, cyclic_press_data, label='Cyclic press Component', color='blue')
# plt.gca().legend_ = None
plt.legend()
plt.title('')
plt.xlabel('')
plt.ylabel('')
# plt.xticks([])
# plt.yticks([])
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
