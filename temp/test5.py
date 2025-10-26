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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_scaled_load_profiles_with_plot():
    # 1. 创建时间索引：2025-10-26（星期日），1分钟频率
    time_index = pd.date_range('2025-10-26', periods=1440, freq='1T')

    # 2. 生成具有真实量纲的原始负荷曲线（单位：MW）
    np.random.seed(42)
    hours = time_index.hour + time_index.minute / 60.0

    # 模拟典型日负荷：基础负荷 + 早晚高峰
    base = 300  # MW
    morning = 400 * np.exp(-((hours - 8.0) ** 2) / (2 * 1.8 ** 2))
    evening = 500 * np.exp(-((hours - 19.5) ** 2) / (2 * 2.2 ** 2))
    noise = 10 * np.random.randn(len(hours))  # 小幅随机扰动

    load_original = base + morning + evening + noise
    load_original = np.clip(load_original, a_min=0, a_max=None)  # 确保非负

    x_max = load_original.max()
    x_min = load_original.min()

    print(f"原始负荷序列（MW）：最小值 = {x_min:.2f}, 最大值 = {x_max:.2f}")

    # 3. 定义目标最小值比例（相对于最大值）
    target_ratios = [0.1, 0.2, 0.3, 0.4]  # 10%, 20%, 30%, 40%

    # 4. 构建 DataFrame
    df = pd.DataFrame(index=time_index)
    df['original_MW'] = load_original

    # 5. 对每个比例进行线性变换（保持形状，调整 min，max 不变）
    for r in target_ratios:
        denom = x_max - x_min
        if denom == 0:
            scaled = load_original.copy()
        else:
            a = (x_max - r * x_max) / denom
            b = x_max - a * x_max
            scaled = a * load_original + b
        col_name = f'min_{int(r*100)}%_MW'
        df[col_name] = scaled

    # 6. 验证结果
    print("\n各列极值验证（单位：MW）：")
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        print(f"{col:>15}: min = {col_min:7.2f}, max = {col_max:7.2f}")

    # 7. 保存到 CSV
    output_file = 'scaled_load_profiles_real_MW.csv'
    df.to_csv(output_file)
    print(f"\n✅ 数据已保存至: {output_file}")

    # 8. 可视化
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['original_MW'], label='Original', linewidth=2, color='black')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, r in enumerate(target_ratios):
        col = f'min_{int(r*100)}%_MW'
        plt.plot(df.index, df[col], label=f'Min = {int(r*100)}% of Max', 
                 linewidth=1.2, color=colors[i])

    plt.title('Scaled Load Profiles (Max Load Fixed, Min Load Adjusted)', fontsize=14)
    plt.xlabel('Time of Day')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.xticks(rotation=0)

    # 设置 x 轴刻度为每 2 小时一个标签
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(13))  # 0,2,4,...,24

    # 保存图像
    plot_file = 'load_profiles_comparison.png'
    plt.savefig(plot_file, dpi=300)
    print(f"✅ 图像已保存至: {plot_file}")

    # 显示图像（在支持的环境中）
    plt.show()

    return df

# 运行主函数
if __name__ == "__main__":
    df_result = generate_scaled_load_profiles_with_plot()
