# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test2.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-26
# * Version     : 1.0.102610
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
import pandas as pd

# 1. 生成一个示例原始时间序列（1天，1分钟频率）
np.random.seed(42)
time_index = pd.date_range('2025-10-26', periods=1440, freq='1T')

# 模拟典型日负荷曲线：早晚高峰，夜间低谷
hours = time_index.hour + time_index.minute / 60
# 构造平滑曲线：双峰（早8点，晚19点）
load = (
    0.3 + 
    0.4 * np.exp(-((hours - 8) ** 2) / (2 * 1.5 ** 2)) +
    0.5 * np.exp(-((hours - 19) ** 2) / (2 * 2.0 ** 2)) +
    0.05 * np.random.randn(len(hours))  # 加少量噪声
)
load = np.clip(load, 0.1, 1.0)  # 确保在合理范围
load = load / load.max()  # 归一化，使最大值为1（便于控制）

x_max = load.max()  # 应为 1.0
x_min = load.min()

# 2. 定义目标最小值比例
ratios = [0.1, 0.2, 0.3, 0.4]

# 3. 生成新序列
df = pd.DataFrame(index=time_index)
df['original'] = load

for r in ratios:
    a = (1 - r) * x_max / (x_max - x_min)
    b = x_max - a * x_max
    new_load = a * load + b
    df[f'min_{int(r*100)}%'] = new_load

# 验证：检查每列的最大值和最小值
print("验证各列极值：")
for col in df.columns:
    print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f}")

# 可选：保存为CSV
# df.to_csv('scaled_load_profiles.csv')




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
