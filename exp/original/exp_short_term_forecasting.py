import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from utils.ts.losses import mape_loss, mase_loss, smape_loss
from utils.ts.m4_summary import M4Summary
from utils.model_tools import EarlyStopping, adjust_learning_rate
from utils.plot_results import test_result_visual

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        # 时间序列模型初始化
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 多 GPU 训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        
        return model 
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader
    
    def _select_criterion(self, loss_name = 'MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.learning_rate
        )

        return model_optim

    def train(self, setting):
        # ------------------------------
        # 数据集构建
        # ------------------------------
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # ------------------------------
        # checkpoint 保存路径
        # ------------------------------
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # ------------------------------
        # 模型训练
        # ------------------------------
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()

            iter_count = 0
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # ------------------------------
                # 模型优化器梯度归零
                # ------------------------------
                model_optim.zero_grad()
                # ------------------------------
                # 数据预处理
                # ------------------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # model forward
                outputs = self.model(batch_x, None, dec_inp, None)
                # 特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                # 预测/实际 label
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                # 计算损失
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())
                # 日志打印：当前 epoch、当前 batch 下每 100 个 batch 的训练速度、误差损失、
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # ------------------------------
                # 后向传播
                # ------------------------------
                loss.backward()
                model_optim.step()
            
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            # 早停机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学习率调整
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        # ------------------------------
        # 最优模型保存、加载
        # ------------------------------
        self.best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(self.best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        # data
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                ).detach().cpu()
            
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)
        self.model.train()
        
        return loss

    def test(self, setting, test = 0):
        # 测试数据集构建
        _, train_loader = self._get_data(flag = 'train')
        _, test_loader = self._get_data(flag = 'test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 最优模型加载
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self.best_model_path))
        # 测试数据结果保存地址
        folder_path = os.path.join(self.args.test_results, setting + '/')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # ------------------------------
        # 模型推理
        # ------------------------------
        self.model.eval()  # 模型推理模式
        with torch.no_grad():
            # dec input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)

            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]], 
                    None
                )

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                test_result_visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)
        # ------------------------------
        # 结果保存 
        # ------------------------------
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pd.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
