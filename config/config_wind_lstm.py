# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_winddata.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-06-11
# * Version     : 0.1.061114
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 单变量-单输出
class Config_Univariate_SingleOutput_V1:
    data_path = "dataset/wind_dataset.csv"
    timestep = 1  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # LSTM 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "LSTM-Univariate-SingleOutput-V1"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# TODO
class Config_Univariate_SingleOutput_V2:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 1  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "LSTM"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# 多变量-单步输出
class Config_MultiVariate_SingleOutput:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = None  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "Config_MultiVariate_SingleOutput"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径


# 多变量-多步输出
class Config_MultiVariate_MultiOutput:
    data_path = "dataset/wind_dataset.csv"
    timestep = 20  # 时间步长，就是利用多少时间窗口
    feature_size = 8  # 每个步长对应的特征数量
    num_layers = 2  # lstm 的层数
    hidden_size = 256  # 隐藏层大小
    output_size = 2  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻的目标值
    split_ratio = 0.8  # 训练测试数据分割比例
    target_index = 0  # 预测特征的列索引
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 1e-4  # 学习率
    best_loss = 0  # 记录损失
    model_name = "TODO"  # 模型名称
    save_path = f"saved_models/{model_name}.pth"  # 最优模型保存路径




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
