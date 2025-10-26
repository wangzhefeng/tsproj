# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
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
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_data():
    """生成示例时间序列数据（一天每分钟的数据）"""
    # 创建时间索引（1440分钟 = 24小时）
    time_index = pd.date_range(start='2023-01-01 00:00:00', 
                              end='2023-01-01 23:59:00', 
                              freq='1min')
    
    # 生成模拟的负荷曲线（典型的日负荷曲线）
    hours = np.arange(1440) / 60  # 转换为小时
    
    # 创建一个典型的日负荷曲线：夜间低谷，白天高峰
    # 使用正弦函数组合模拟真实负荷曲线
    base_load = 50
    peak_load = 100
    
    # 主要的日周期（24小时）
    daily_cycle = np.sin(2 * np.pi * (hours - 6) / 24)  # 峰值在下午2点左右
    
    # 添加一些随机噪声
    noise = np.random.normal(0, 5, 1440)
    
    # 组合负荷曲线
    load_data = base_load + peak_load * (daily_cycle + 1) / 2 + noise
    
    # 确保所有值为正数
    load_data = np.maximum(load_data, 10)
    
    return pd.Series(load_data, index=time_index)

def process_time_series(original_data, min_ratio):
    """
    处理时间序列数据
    
    参数:
    original_data: 原始时间序列数据
    min_ratio: 最小值占最大值的比例（0.1, 0.2, 0.3, 0.4）
    
    返回:
    processed_data: 处理后的时间序列数据
    """
    # 获取原始数据的最大值和最小值
    original_max = original_data.max()
    original_min = original_data.min()
    
    # 新的最大值保持不变
    new_max = original_max
    
    # 新的最小值为最大值的指定比例
    new_min = new_max * min_ratio
    
    # 保持曲线形状不变的关键：线性变换
    # 对于原始数据中的每个点 x，映射到新数据中的点 y
    # y = new_min + (x - original_min) * (new_max - new_min) / (original_max - original_min)
    
    if original_max == original_min:
        # 如果所有值都相同，直接返回新最小值（也是最大值）
        processed_data = pd.Series([new_min] * len(original_data), index=original_data.index)
    else:
        processed_data = new_min + (original_data - original_min) * (new_max - new_min) / (original_max - original_min)
    
    return processed_data

def main():
    # 1. 生成示例数据
    print("生成原始时间序列数据...")
    original_data = generate_sample_data()
    
    print(f"原始数据统计:")
    print(f"  数据点数量: {len(original_data)}")
    print(f"  最大值: {original_data.max():.2f}")
    print(f"  最小值: {original_data.min():.2f}")
    print(f"  最小值/最大值比例: {original_data.min()/original_data.max():.2%}")
    
    # 2. 定义要处理的最小值比例
    min_ratios = [0.1, 0.2, 0.3, 0.4]
    
    # 3. 处理数据
    processed_data_dict = {}
    for ratio in min_ratios:
        processed_data = process_time_series(original_data, ratio)
        processed_data_dict[ratio] = processed_data
        
        print(f"\n处理结果 (最小值 = 最大值的 {ratio*100:.0f}%):")
        print(f"  最大值: {processed_data.max():.2f} (保持不变)")
        print(f"  最小值: {processed_data.min():.2f}")
        print(f"  最小值/最大值比例: {processed_data.min()/processed_data.max():.2%}")
    
    # 4. 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 原始数据
    plt.subplot(2, 3, 1)
    plt.plot(original_data.index, original_data.values, 'b-', alpha=0.7)
    plt.title('原始数据')
    plt.xlabel('时间')
    plt.ylabel('负荷值')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 处理后的数据
    subplot_positions = [2, 3, 4, 5, 6]
    for i, ratio in enumerate(min_ratios):
        plt.subplot(2, 3, i+2)
        plt.plot(processed_data_dict[ratio].index, processed_data_dict[ratio].values, 'r-', alpha=0.7)
        plt.title(f'最小值 = 最大值的 {ratio*100:.0f}%')
        plt.xlabel('时间')
        plt.ylabel('负荷值')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 验证曲线形状保持不变
    print("\n验证曲线形状保持:")
    print("通过计算相邻点的变化率来验证形状是否保持不变")
    
    # 选择一个比例进行验证（比如0.2）
    test_ratio = 0.2
    original_diff = np.diff(original_data.values)
    processed_diff = np.diff(processed_data_dict[test_ratio].values)
    
    # 计算变化率的比例
    scaling_factor = (processed_data_dict[test_ratio].max() - processed_data_dict[test_ratio].min()) / \
                     (original_data.max() - original_data.min())
    
    print(f"理论缩放因子: {scaling_factor:.4f}")
    
    # 检查实际缩放是否一致（忽略很小的数值以避免除零错误）
    valid_indices = np.abs(original_diff) > 1e-10
    if np.any(valid_indices):
        actual_ratios = processed_diff[valid_indices] / original_diff[valid_indices]
        print(f"实际变化率缩放因子范围: [{actual_ratios.min():.4f}, {actual_ratios.max():.4f}]")
        print(f"平均实际缩放因子: {actual_ratios.mean():.4f}")
        print("如果理论值和实际值接近，说明曲线形状确实保持不变")
    
    return original_data, processed_data_dict

# 运行主函数
if __name__ == "__main__":
    original_data, processed_results = main()
