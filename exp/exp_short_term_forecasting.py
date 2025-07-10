# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_short_term_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.model_tools import EarlyStopping, adjust_learning_rate
from utils.ts.losses import mape_loss, mase_loss, smape_loss
from utils.ts.m4 import M4Meta
from utils.ts.m4_summary import M4Summary
from utils.model_memory import model_memory_size
from utils.plot_results import predict_result_visual
from utils.plot_losses import plot_losses
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger


class Exp_Short_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        # 时间序列模型初始化
        model = self.model_dict[self.args.model].Model(self.args)
        # 多 GPU 训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        # 打印模型参数量
        total_memory_gb = model_memory_size(model, verbose=True)
        
        return model 
    
    def _get_data(self, flag: str):
        """
        数据集构建
        """
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader
    
    def _select_criterion(self):
        """
        评价指标
        """
        if self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAPE':
            return mape_loss()
        elif self.args.loss == 'MASE':
            return mase_loss()
        elif self.args.loss == 'SMAPE':
            return smape_loss()

    def _select_optimizer(self):
        """
        优化器
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.learning_rate
        )

        return optimizer

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = os.path.join(model_path, "checkpoint.pth")
        
        return model_checkpoint_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting)
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _get_predict_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.predict_results, setting)
        os.makedirs(results_path, exist_ok=True)
        
        return results_path 

    # TODO
    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
        from utils.ts.metrics_dl import metric, DTW
        # 计算测试结果评价指标
        mse, rmse, mae, mape, mape_accuracy, mspe = metric(preds, trues)
        dtw = DTW(preds, trues) if self.args.use_dtw else -999
        logger.info(f"Test results: mse:{mse:.4f} rmse:{rmse:.4f} mae:{mae:.4f} mape:{mape:.4f} mape accuracy:{mape_accuracy:.4f} mspe:{mspe:.4f} dtw: {dtw:.4f}")
        
        # result1 保存
        with open(os.path.join(path, "result_forecast.txt"), 'a') as file:
            file.write(setting + "  \n")
            file.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, mape accuracy:{mape_accuracy}, mspe:{mspe}, dtw:{dtw}")
            file.write('\n')
            file.write('\n')
            file.close()
        # result2 保存
        np.save(
            os.path.join(path, 'metrics.npy'), 
            np.array([mae, mse, rmse, mape, mape_accuracy, mspe, dtw])
        )
        np.save(os.path.join(path, 'preds.npy'), preds)
        np.save(os.path.join(path, 'trues.npy'), trues)
    
    def _pred_results_save(self, preds, preds_df, path):
        """
        预测结果保存
        """
        if preds is not None:
            np.save(os.path.join(path, "prediction.npy"), preds) 
        if preds_df is not None:
            preds_df.to_csv(
                os.path.join(path, "prediction.csv"), 
                encoding="utf_8_sig", 
                index=False
            )

    def train(self, setting):
        # 数据集构建
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid')
        # checkpoint 保存路径
        logger.info(f"{40 * '-'}")
        logger.info(f"Model checkpoint will be saved in path:")
        logger.info(f"{40 * '-'}")
        model_checkpoint_path = self._get_model_path(setting)
        logger.info(model_checkpoint_path)
        # 测试结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Train results will be saved in path:")
        logger.info(f"{40 * '-'}")
        test_results_path = self._get_test_results_path(setting) 
        logger.info(test_results_path)
        # 模型训练
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start training...")
        logger.info(f"{40 * '-'}")
        # time: 模型训练开始时间
        train_start_time = time.time()
        logger.info(f"Train start time: {from_unix_time(train_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        # 训练窗口数
        train_steps = len(train_loader)
        logger.info(f"Train total steps: {train_steps}") 
        # 模型优化器
        optimizer = self._select_optimizer()
        logger.info(f"Train optimizer has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
        # TODO MSE 损失函数
        # mse = nn.MSELoss()
        # 自动混合精度训练
        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        # 训练、验证结果收集
        train_losses, vali_losses = [], []
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 训练开始时间
            epoch_start_time = time.time()
            logger.info(f"Epoch: {epoch+1} \tstart time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            # epoch 训练结果收集
            iter_count = 0
            train_loss = []
            # 模型训练模式
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # logger.info(f"Train step: {i} running...")
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # ------------------------------
                # 前向传播
                # ------------------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # model forward
                outputs = self.model(batch_x, None, dec_inp, None)
                # 预测/实际 label
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                # 计算损失
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())
                
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 5 == 0:
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
                    iter_count = 0
                    train_start_time = time.time()
                # ------------------------------
                # 后向传播、参数优化更新
                # ------------------------------
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")
            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss = self.valid(train_loader, vali_loader, criterion)
            logger.info(f"Epoch: {epoch + 1}, \tSteps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            # 训练/验证损失收集
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            # 早停机制、模型保存
            early_stopping(
                vali_loss, 
                epoch=epoch, 
                model=self.model, 
                optimizer=optimizer, 
                scheduler=None, 
                model_path=model_checkpoint_path,
            )
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, \tEarly stopping...")
                break
            # 学习率调整
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        # -----------------------------
        # 模型加载
        # ------------------------------
        logger.info(f"{40 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{40 * '-'}")
        # plot losses
        logger.info("Plot and save train/valid losses...")
        plot_losses(
            train_epochs=self.args.train_epochs,
            train_losses=train_losses, 
            vali_losses=vali_losses, 
            label="loss",
            results_path=test_results_path
        )
        # load model
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, train_loader, vali_loader, criterion):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"Model start validating...")
        # TODO 数据处理
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 模型验证结果
        # vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            logger.info(f"Valid step: running...")
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()#.to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            # 前向传播
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                ).detach().cpu()
            # 预测值/真实值提取
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)
            # 计算/保存验证损失
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)
            # vali_loss.append(loss)
            # logger.info(f"debug::valid step: {i}, valid loss: {loss.item()}")
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"Validating Finished!")
        return loss

    def test(self, setting, load: bool=False):
        # 数据集构建
        _, train_loader = self._get_data(flag = 'train')
        _, test_loader = self._get_data(flag = 'test')
        # TODO 数据处理
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"]) 
            logger.info(model_checkpoint_path)
        # 测试结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Test results will be saved in path:")
        logger.info(f"{40 * '-'}")
        test_results_path = self._get_test_results_path(setting) 
        logger.info(test_results_path) 
        # 模型开始测试
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start testing...")
        logger.info(f"{40 * '-'}")
        # ------------------------------
        # 模型推理
        # ------------------------------
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], []
        with torch.no_grad():
            logger.info(f"Test step: running...")
            # dec input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            # 前向传播
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]], 
                    None
                )
                if id_list[i] % 1000 == 0:
                    logger.info(f"id_list[i]: {id_list[i]}")
            # 预测值/真实值提取
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()
            # 测试结果收集
            preds = outputs
            trues = y
            # 预测数据可视化
            x = x.detach().cpu().numpy()
            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                true_plot = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pred_plot = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                predict_result_visual(pred_plot, true_plot, path = os.path.join(test_results_path, str(i) + '.pdf'))
        # 测试结果保存
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        forecasts_df = pd.DataFrame(
            preds[:, :, 0], 
            columns=[f'V{i + 1}' for i in range(self.args.pred_len)]
        )
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(test_results_path + folder_path + self.args.seasonal_patterns + '_forecast.csv')
        logger.info(test_results_path)
        
        if ('Weekly_forecast.csv' in os.listdir(folder_path) \
            and 'Monthly_forecast.csv' in os.listdir(folder_path) \
            and 'Yearly_forecast.csv' in os.listdir(folder_path) \
            and 'Daily_forecast.csv' in os.listdir(folder_path) \
            and 'Hourly_forecast.csv' in os.listdir(folder_path) \
            and 'Quarterly_forecast.csv' in os.listdir(folder_path)
        ):
            m4_summary = M4Summary(folder_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            logger.info(f"Test results: smape:{smape_results:.4f} mape:{mape:.4f} mase:{mase:.4f} owa:{owa_results:.4f}")
        else:
            logger.info('After all 6 tasks are finished, you can calculate the averaged index')
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
