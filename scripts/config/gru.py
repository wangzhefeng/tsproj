# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_wind_gru.py
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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Config:
    model_name = "GRU"  # 模型名称
    data = "wind"
    features = "MS"
    pred_method = "recursive_multi_step"  # "recursive_multi_step", "direct_multi_step_output", "direct_recursive_mix"
    if sys.platform == "win32":
        data_path = "E:/projects/ts_projects/tsproj/dataset/wind_dataset.csv"
    else:
        data_path = "/Users/wangzf/tsproj/dataset/wind_dataset.csv"
    target = "WIND"
    target_index = 0
    split_ratio = 0.8  # 训练数据数据分割比例
    seq_len = 4  # 时间步长，就是利用多少时间窗口(lags)
    feature_size = 3  # 每个步长对应的特征数量
    hidden_size = 256  # GRU 隐藏层大小
    num_layers = 2  # GRU 的层数
    output_size = 1  # 由于是单输出任务，最终输出层大小为 1，预测未来 1 个时刻数据
    epochs = 10  # 迭代轮数
    batch_size = 32  # 批次大小
    learning_rate = 3e-4  # 学习率
    loss_name = "MSE"
    best_loss = 0.1 # 记录损失
    scale = False
    use_gpu = False
    use_multi_gpu = False
    gpu = "0"
    devices = "cuda:0"
    device_ids = "0,1,2,3"
    if sys.platform == "win32":
        save_path = f"E:/projects/timeseries_forecasting/tsproj/tsproj_dl/saved_models/{model_name}-sl{seq_len}-fz{feature_size}-nl{num_layers}-os{output_size}.pth"
    else:
        save_path = f"/Users/wangzf/tsproj/tsproj_dl/saved_models/{model_name}-sl{seq_len}-fz{feature_size}-nl{num_layers}-os{output_size}.pth"


class Config_test:
    data_path = None
    # data
    import pandas as pd
    data = pd.DataFrame({
        "Date": pd.to_datetime([
            "1961-01-01", "1961-01-02", "1961-01-03", "1961-01-04", "1961-01-05",
            "1961-01-06", "1961-01-07", "1961-01-08", "1961-01-09", "1961-01-10",
        ]),
        "Wind": [13.67, 11.50, 11.25, 8.63, 11.92, 12.3, 11.5, 13.22, 11.77, 10.51],
        "Temperature": [12, 18, 13, 27, 5, 12, 15, 20, 22, 13],
        "Rain": [134, 234, 157, 192, 260, 167, 281, 120, 111, 223],
    })
    data.set_index("Date", inplace = True)
    # print(data)
    # print(data.values)
    # print(data.shape)
    # print("-" * 80)
    data = data.values
    target = "WIND"
    target_index = 0
    split_ratio = 0.8
    seq_len = 2
    feature_size = 3
    output_size = 11
    batch_size = 32
    features="MS"
    pred_method = "recursive_multi_step"




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()