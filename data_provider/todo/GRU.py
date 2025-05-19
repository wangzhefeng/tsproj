# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_splitor.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-20
# * Version     : 1.0.012021
# * Description : https://blog.csdn.net/java1314777/article/details/134407174
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def create_input_sequences(input_data, window_len: int, predict_len: int, step_size: int = 1):
    """
    创建时间序列数据专用的数据分割器

    Args:
        input_data (_type_): 输入数据
        window_len (int): 窗口大小
        predict_len (int): 预测长度
        step_size (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    output_seq = []
    input_data_len = len(input_data)
    for i in range(0, input_data_len - window_len, step_size):
        train_seq = input_data[i:(i + window_len)]
        if (i + window_len + predict_len) > len(input_data):
            break
        train_label = input_data[(i + window_len):(i + window_len + predict_len)]
        output_seq.append((train_seq, train_label))
    
    # 样本数量
    sample_num = input_data_len - (window_len + predict_len - 1)
    logger.info(f"sample number: {sample_num}")
    
    return output_seq


class TimeSeriesDataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        
        return torch.Tensor(sequence), torch.Tensor(label)


class GRU(nn.Module):
    
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1, pred_len= 4):
        super(GRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0_gru = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.gru(x, h0_gru)
        out = self.dropout(out)
        
        # 取最后 pred_len 时间步的输出
        out = out[:, -self.pred_len:, :]
        
        out = self.fc(out)
        out = self.relu(out)
        
        return out
 
 
def calculate_mae(y_true, y_pred):
    """
    平均绝对误差
    """
    mae = np.mean(np.abs(y_true - y_pred))
    
    return mae
 



# 测试代码 main 函数
def main():
    # params
    data_path = "./dataset/ETT-small/ETTh1.csv"
    target = "OT"
    window_len = 6  # 窗口长度
    predict_len = 2  # 预测长度
    batch_size = 1
    train_size = 0.80
    test_size = 0.20
    epochs = 10
    training = False
    model_path = "./tsproj_dl/saved_models/save_model.pth"
    test_results_path = "./tsproj_dl/saved_models/test_results.png"
    
    # random seed
    np.random.seed(10)
    
    # data
    data = pd.read_csv(data_path)
    logger.info(f"data: \n{data.head()}")
    logger.info(f"data: \n{data.shape}")
    data = np.array(data[target])
    logger.info(f"data: \n{data}")
    logger.info(f"data length: \n{len(data)}")
    
    # split data
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx].reshape(-1, 1)
    test_data = data[split_idx:].reshape(-1, 1)
    logger.info(f"train_data.shape: {train_data.shape}")
    logger.info(f"test_data.shape: {test_data.shape}")
    
    # transform data
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    train_data_normalized = scaler_train.fit_transform(train_data)
    test_data_normalized = scaler_test.fit_transform(test_data)
    
    # numpy to tensor
    train_data_tensor = torch.FloatTensor(train_data_normalized)
    test_data_tensor = torch.FloatTensor(test_data_normalized)
    
    # 创建 dataset
    train_dataset = create_input_sequences(train_data_tensor, window_len, predict_len, step_size=1)
    test_dataset = create_input_sequences(test_data_tensor, window_len, predict_len, step_size=1)
    
    # 创建 Dataset
    train_dataset = TimeSeriesDataset(train_dataset)
    test_dataset = TimeSeriesDataset(test_dataset)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        dataset = train_dataset, 
        batch_size = batch_size, 
        shuffle = True,
        drop_last = True,
    )
    test_loader = DataLoader(
        dataset = test_dataset, 
        batch_size = batch_size, 
        shuffle = False,
        drop_last = True
    )
    logger.info(f"length train_loader: {len(train_loader)}")
    logger.info(f"length test_loader: {len(test_loader)}")
   
    # tese
    # for seq, labels in train_loader:
    #     logger.info(seq.shape)
    #     logger.info(f"seq: \n{seq}")
    #     logger.info(labels.shape)
    #     logger.info(f"labels: \n{labels}")
    #     break
    
    # model
    lstm_model = GRU(
        input_dim=1, 
        output_dim=1, 
        num_layers=2, 
        hidden_dim=window_len, 
        pred_len=predict_len
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
    
    # model training
    if training:
        losss = []
        # 训练模式
        lstm_model.train()
        for i in range(epochs):
            start_time = time.time()
            for seq, labels in train_loader:
                lstm_model.train()
                optimizer.zero_grad()
                y_pred = lstm_model(seq)
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
                logger.info(f'epoch: {i}/{epochs} loss: {single_loss.item():10.8f}')
            losss.append(single_loss.detach().numpy())
        torch.save(lstm_model.state_dict(), model_path)
        logger.info(f"模型已保存, 用时:{(time.time() - start_time) / 60:.4f}min")
    else:
        # 加载模型进行预测
        lstm_model.load_state_dict(torch.load(model_path))
        # 评估模式
        lstm_model.eval()
        # 测试结果输出
        trues = []
        preds = []
        losses = []
        for batch, (seq, labels) in enumerate(test_loader):
            logger.info(f"seq.shape: {seq.shape}")
            # logger.info(f"seq: \n{seq}")
            logger.info(f"labels.shape: {labels.shape}")
            # logger.info(f"labels: \n{labels}")
            
            # 前向传播
            pred = lstm_model(seq)
            logger.info(f"pred.shape: {pred.shape}")
            # logger.info(f"pred: \n{pred}")
            
            # 计算误差
            mae = calculate_mae(pred.detach().numpy(), np.array(labels))
            logger.info(f"mae: {mae}")
            losses.append(mae)
            
            # 结果处理
            for j in range(batch_size):
                for i in range(predict_len):
                    trues.append(labels[j][i][0].detach().numpy())
                    preds.append(pred[j][i][0].detach().numpy())
            if batch == 1:
                break
        # 测试结果
        trues = scaler_test.inverse_transform(np.array(trues).reshape(1, -1))[0]
        preds = scaler_test.inverse_transform(np.array(preds).reshape(1, -1))[0]
        logger.info(f"模型真实标签: \n{trues}, \ntrues length: {len(trues)}")
        logger.info(f"模型预测结果: \n{preds}, \npreds length: {len(preds)}")
        logger.info(f"预测误差MAE: \n{losses}, \nlosses length: {len(losses)}")
        # 结果可视化
        fig = plt.figure(figsize = (15, 8))
        plt.plot(trues, label='trues', color='blue')
        plt.plot(preds, label='preds', color='red', linestyle='--')
        plt.title('real vs forecast')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid()
        plt.savefig(test_results_path)
        plt.show()

if __name__ == "__main__":
    main()
