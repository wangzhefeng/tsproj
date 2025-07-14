# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_winddata_cnn_attn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-06-11
# * Version     : 0.1.061115
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Config:
    data_path = "data/wind_data.csv"
    seq_len = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_heads = 1  # 注意力机制头的数量
    out_channels = [10, 20, 30]  # 卷积层输出通道
    # num_layers = 2  # 网络的层数
    # hidden_size = 256  # 网络隐藏层大小
    output_size = 1  # 预测未来 n 个时刻数据
    epochs = 50  # 迭代轮数
    batch_size = 16  # 批次大小
    learning_rate = 1e-5  # 学习率
    best_loss = 0  # 记录损失
    model_name = "CNN-Attention"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()