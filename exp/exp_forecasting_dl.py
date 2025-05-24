# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052710
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
# data pipeline
from data_provider.data_factory_dl_1 import data_provider
# model training
from utils.model_tools import adjust_learning_rate, EarlyStopping
# loss
from utils.losses import mape_loss, mase_loss, smape_loss
# metrics
from utils.metrics_dl import metric, DTW
from utils.plot_results import test_result_visual
from utils.plot_losses import plot_losses
# log
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model].Model(self.args)
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.devices)
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of model parameters: {(total / 1e6):.2f}M')
        
        return model
    
    def _build_multi_model(self):
        """
        多个模型构建
        """
        # 模型列表
        models = []
        for i in range(self.args.output):
            model = self._build_model()
            # 模型收集
            models.append(model)
        
        return models
    
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
        if self.args.loss == "MSE":
            return nn.MSELoss()
        elif self.args.loss == "MAPE":
            return mape_loss()
        elif self.args.loss == "MASE":
            return mase_loss()
        elif self.args.loss == "SMAPE":
            return smape_loss()

    def _select_optimizer(self):
        """
        优化器
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = self.args.learning_rate
        )
        
        return optimizer
    
    def _select_multip_optimizer(self):
        """
        优化器
        """
        optimizers = [
            self._select_optimizer()
            for i in range(self.args.output)
        ]
        
        return optimizers
    
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

    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
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

    def save_model(self, weights: bool = False):
        """
        模型保存
        """
        logger.info(f'Model saved in {self.args.checkpoints}')
        if weights:
            # model weights
            torch.save(self.model.state_dict(), self.args.checkpoints)
        else:
            # whole model
            torch.save(self.model, self.args.checkpoints)

    def load_model(self, weights: str = False):
        logger.info(f'Model Loading model from {self.args.checkpoints}')
        if weights:
            self.model = self.model_dict[self.args.model].Model(self.args)
            self.model.load_state_dict(torch.load(self.args.checkpoints))
        else:
            self.model = torch.load(self.args.checkpoints)
        self.model.eval()

    def _inverse_data(self, data, outputs, batch_y):
        """
        输入输出逆转换
        """
        if data.scale and self.args.inverse:
            outputs = data.inverse_target(outputs)
            batch_y = data.inverse_target(batch_y)
        logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
        logger.info(f"debug::batch_y: \n{batch_y} \nbatch_y.shape: {batch_y.shape}")
        
        return outputs, batch_y

    def train(self, setting):
        """
        模型训练
        """
        # 数据集构建
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
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
        logger.info(f"Train steps: {train_steps}") 
        # 模型优化器
        optimizer = self._select_optimizer()
        logger.info(f"Train optimizer has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
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
            for i, data_batch in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 取出数据
                x_train, y_train = data_batch
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 前向传播
                outputs = self.model(x_train)
                # TODO 输入输出逆转换
                # outputs, y_train = self._inverse_data(train_data, outputs, y_train)
                # TODO 预测值/真实值提取
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:].to(self.device)
                # y_train = y_train[:, :, f_dim:].to(self.device)
                # logger.info(f"debug::outputs: \n{outputs}, \noutputs.shape: {outputs.shape}")
                # logger.info(f"debug::batch_y: \n{batch_y}, \nbatch_y.shape: {batch_y.shape}")
                # 计算训练损失
                if self.args.output_size == 1:
                    loss = criterion(outputs, y_train.reshape(-1, 1))
                else:
                    loss = criterion(outputs, y_train)
                train_loss.append(loss.item())
                logger.info(f"debug::train step: {i}, train loss: {loss.item()}")
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 10 == 0:
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
                    iter_count = 0
                    train_start_time = time.time()
                # 后向传播、参数优化更新
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")
            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
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
        logger.info("Plot and save train/vali losses...")
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

    def train_multi_model(self, setting):
        """
        模型训练
        """
        # 数据集构建
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
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
        logger.info(f"Train steps: {train_steps}") 
        # 模型优化器
        optimizers = self._select_multip_optimizer()
        logger.info(f"Train optimizers has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
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
            models = self._build_multi_model()
            for model_idx, model in enumerate(models):
                # epoch 训练结果收集
                iter_count = 0
                train_loss = []
                # 优化器
                optimizer = optimizers[model_idx]
                # 模型训练模式
                self.model.train()
                for i, data_batch in enumerate(train_loader):
                    # 当前 epoch 的迭代次数记录
                    iter_count += 1
                    # 取出数据
                    x_train, y_train = data_batch
                    # 模型优化器梯度归零
                    optimizer.zero_grad()
                    # 前向传播
                    outputs = self.model(x_train)
                    # TODO 输入输出逆转换
                    # outputs, y_train = self._inverse_data(train_data, outputs, y_train)
                    # TODO 预测值/真实值提取
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, :, f_dim:].to(self.device)
                    # y_train = y_train[:, :, f_dim:].to(self.device)
                    # logger.info(f"debug::outputs: \n{outputs}, \noutputs.shape: {outputs.shape}")
                    # logger.info(f"debug::batch_y: \n{batch_y}, \nbatch_y.shape: {batch_y.shape}")
                    # 计算训练损失
                    if self.args.output_size == 1:
                        loss = criterion(outputs, y_train[:, model_idx].reshape(-1, 1))
                    else:
                        loss = criterion(outputs, y_train[:, model_idx])
                    train_loss.append(loss.item())
                    logger.info(f"debug::train step: {i}, train loss: {loss.item()}")
                    # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                    if (i + 1) % 10 == 0:
                        speed = (time.time() - train_start_time) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
                        iter_count = 0
                        train_start_time = time.time()
                    # 后向传播、参数优化更新
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")
                # 模型验证
                train_loss = np.average(train_loss)
                vali_loss = self.vali_multi_model(vali_loader, criterion, model_idx)
                logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
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
        '''
        # -----------------------------
        # 模型加载
        # ------------------------------
        logger.info(f"{40 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{40 * '-'}")
        # plot losses
        logger.info("Plot and save train/vali losses...")
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
        '''
        
        return None

    def vali(self, vali_loader, criterion):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start validating...")
        logger.info(f"{40 * '-'}")
        # 验证窗口数
        vali_steps = len(vali_loader)
        logger.info(f"Vali steps: {vali_steps}")
        # 模型验证结果
        vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                x_vali, y_vali = data_batch
                # 前向传播
                outputs = self.model(x_vali)
                # 计算/保存验证损失
                if self.args.output_size == 1:
                    loss = criterion(outputs, y_vali.reshape(-1, 1))
                else:
                    loss = criterion(outputs, y_vali)
                vali_loss.append(loss.item())
                logger.info(f"debug::vali step: {i}, vali loss: {loss}")
        # 计算验证集上所有 batch 的平均验证损失
        vali_loss = np.average(vali_loss)
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Validating Finished!")
        logger.info(f"{40 * '-'}")
        
        return vali_loss

    def vali_multi_model(self, vali_loader, criterion, model_idx):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start validating...")
        logger.info(f"{40 * '-'}")
        # 验证窗口数
        vali_steps = len(vali_loader)
        logger.info(f"Vali steps: {vali_steps}")
        # 模型验证结果
        vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                x_vali, y_vali = data_batch
                # 前向传播
                outputs = self.model(x_vali)
                # 计算/保存验证损失
                if self.args.output_size == 1:
                    loss = criterion(outputs, y_vali[:, model_idx].reshape(-1, 1))
                else:
                    loss = criterion(outputs, y_vali[:, model_idx])
                vali_loss.append(loss.item())
                logger.info(f"debug::vali step: {i}, vali loss: {loss}")
        # 计算验证集上所有 batch 的平均验证损失
        vali_loss = np.average(vali_loss)
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Validating Finished!")
        logger.info(f"{40 * '-'}")
        
        return vali_loss

    def test(self, flag, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag=flag) 
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
        # 模型测试次数
        test_steps = len(test_loader)
        logger.info(f"Test steps: {test_steps}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], []
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                logger.info(f"test step: {i}")
                x_test, y_test = data_batch
                # 前向传播
                outputs = self.model(x_test)
                # TODO 输入输出逆转换
                # outputs, y_test = self._inverse_data(test_data, outputs, y_test)
                # TODO 预测值/真实值提取
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:]
                # y_test = y_test[:, :, f_dim:]
                # logger.info(f"debug::pred: \n{pred} \npred shape: {pred.shape}")
                # logger.info(f"debug::true: \n{true} \ntrue shape: {true.shape}")
                # 验证结果收集
                pred = outputs
                true = y_test
                preds.append(pred)
                trues.append(true)
                # TODO test batch_size > 1
                # if test_loader.batch_size > 1:
                #     for batch_idx in range(self.args.batch_size):
                #         preds_flat.append(pred[batch_idx, :, -1].tolist())
                #         trues_flat.append(true[batch_idx, :, -1].tolist())
                #     logger.info(f"debug::preds_flat: \n{preds_flat} \npreds_flat length: {len(preds_flat)}")
                #     logger.info(f"debug::trues_flat: \n{trues_flat} \ntrues_flat length: {len(trues_flat)}")
                # 预测数据可视化
                if i % 5 == 0:
                    inputs = x_test.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        # or
                        # inputs = test_data.inverse_transform(inputs.squeeze(0)).reshape(shape)
                    pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    true_plot = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    test_result_visual(pred_plot, true_plot, path = os.path.join(test_results_path, str(i) + '.pdf')) 
        # 测试结果保存
        # preds = np.array(preds)  # or 
        preds = np.concatenate(preds, axis = 0)
        # trues = np.array(trues)  # or 
        trues = np.concatenate(trues, axis = 0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # logger.info(f"Test results: preds: \n{preds} \npreds.shape: {preds.shape}")
        # logger.info(f"Test results: trues: \n{trues} \ntrues.shape: {trues.shape}")
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._test_results_save(preds, trues, setting, test_results_path)
        logger.info(test_results_path)
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        # TODO test batch_size > 1
        # if test_loader.batch_size > 1:
        #     preds_flat = np.concatenate(preds_flat, axis = 0)
        #     trues_flat = np.concatenate(trues_flat, axis = 0)
        preds_flat = np.concatenate(preds, axis = 0)
        trues_flat = np.concatenate(trues, axis = 0)
        test_result_visual(preds_flat, trues_flat, path = os.path.join(test_results_path, "test_prediction.png")) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    def test_v3(self, flag, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag=flag) 
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
        # 模型测试次数
        test_steps = len(test_loader)
        logger.info(f"Test steps: {test_steps}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], []
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                logger.info(f"test step: {i}")
                x_test, y_test = data_batch
                # 前向传播
                outputs = self.model(x_test)
                from utils.metrics_dl import MAE
                mae = MAE(outputs.detach().numpy(), np.array(y_test))
                logger.info(f"MAE: {mae:.4f}")
                # 验证结果收集
                for j in range(self.args.batch_size):
                    for i in range(self.predict_len):
                        preds.append(outputs[j][i][0].detach().numpy())
                        trues.append(y_test[j][i][0].detach().numpy()) 
                if i == 1:
                    break 
        # 测试结果保存
        preds = np.array(preds).reshape(1, -1)
        trues = np.array(trues).reshape(1, -1)
        preds = test_data.inverse_transform(preds)
        trues = test_data.inverse_transform(trues)
        logger.info(f"Test results: preds: \n{preds} \npreds.shape: {preds.shape}")
        logger.info(f"Test results: trues: \n{trues} \ntrues.shape: {trues.shape}")
        
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._test_results_save(preds, trues, setting, test_results_path)
        logger.info(test_results_path)
        
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        preds_flat = np.concatenate(preds, axis = 0)
        trues_flat = np.concatenate(trues, axis = 0)
        test_result_visual(preds_flat, trues_flat, path = os.path.join(test_results_path, "test_prediction.png")) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    def forecast_single_step(self, data):
        """
        单步预测
        """
        logger.info('Model Predicting Point-by-Point...')
        pred = self.model.predict(data)
        pred = np.reshape(pred, (pred.size,))
        
        return pred

    def forecast_direct_multi_output(self, plot_size):
        # data load
        data_loader, _, _ = self._get_data()
        # train result
        y_train_pred = data_loader.scaler.inverse_transform(
            (self.model(data_loader.x_train_tensor).detach().numpy()[:plot_size]).reshape(-1, 1)
        )
        y_train_true = data_loader.scaler.inverse_transform(
            data_loader.y_train_tensor.detach().numpy().reshape(-1, 1)[:plot_size]
        )
        # test result
        y_test_pred = data_loader.scaler.inverse_transform(
            self.model(data_loader.x_test_tensor).detach().numpy()[:plot_size]
        )
        y_test_true = data_loader.scaler.inverse_transform(
            data_loader.y_test_tensor.detach().numpy().reshape(-1, 1)[:plot_size]
        )

        return (y_train_pred, y_train_true), (y_test_pred, y_test_true)

    def forecast_direct_multi_step(self, data):
        pass

    def forecast_recursive_multi_step(self, data, window_size: int, horizon: int):
        """
        时序多步预测
            - 每次预测使用 window_size 个历史数据进行预测，预测未来 prediction_len 个预测值
            - 每一步预测一个点，然后下一步将预测的点作为历史数据进行下一次预测

        Args:
            data (_type_): 测试数据
            window_size (int): 窗口长度
            prediction_len (int): 预测序列长度

        Returns:
            _type_: 预测序列
        """
        logger.info('ModelPredicting Sequences Multiple...')
        preds_seq = []  # (20, 50, 1)
        for i in range(int(len(data) / horizon)):  # 951 / 50 = 19
            curr_frame = data[i * horizon]  # (49, 1)
            preds = []  # 50
            for j in range(horizon):  # 50
                pred = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]  # curr_frame[newaxis, :, :].shape: (1, 49, 1) => (1,)
                preds.append(pred)
                curr_frame = curr_frame[1:]  # (48, 1)
                curr_frame = np.insert(curr_frame, [window_size - 2], preds[-1], axis = 0)
            preds_seq.append(preds)
        
        return preds_seq
    
    def forecast_recursive_hybird(self, data):
        pass

    def forecast_seq2seq_multi_step(self, data):
        pass
    
    def forecast_sequence_full(self, data, window_size: int):
        """
        单步预测

        Args:
            data (_type_): 测试数据
            window_size (_type_): 窗口长度

        Returns:
            _type_: 预测序列
        """
        logger.info('ModelPredicting Sequences Full...')
        curr_frame = data[0]
        preds_seq = []
        for i in range(len(data)):
            pred = self.model.predict(curr_frame[np.newaxis, :, :])[0, 0]
            preds_seq.append(pred)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], preds_seq[-1], axis = 0)
        
        return preds_seq

    @staticmethod
    def plot_train_results(pred, true, task = "Train"):
        plt.figure(figsize = (15, 8))
        plt.plot(pred, label = "Pred")
        plt.plot(true, label = "True")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"{task} Predictions")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show();




# 测试代码 main 函数
def main():
    from utils.tsproj_dl.config.gru import Config
    # config
    configs = Config()
    exp = Exp_Forecast(args=configs)
    # model train
    exp.train()
    # model predict
    (y_train_pred, y_train_true), (y_test_pred, y_test_true) = exp.predict_directly_multi_output(plot_size = 200)
    logger.info(f"y_train_pred: \n{y_train_pred}")
    logger.info(f"y_test_pred: \n{y_test_pred}")
    # result plot
    exp.plot_train_results(y_train_pred, y_train_true, task="Train")
    exp.plot_train_results(y_test_pred, y_test_true, task="Test")
    '''
    # ------------------------------
    # 模型测试
    # ------------------------------
    model = None
    # multi-sequence
    # --------------
    predictions_multiseq = model.predict_sequences_multiple(
        data = x_test, # shape: (656, 49, 1)
        window_size = configs['data']['sequence_length'],  # 50
        prediction_len = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(predictions_multiseq).shape)
    plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'], title = configs.data)
    
    # point by point
    # --------------
    predictions_pointbypoint = model.predict_point_by_point(data = x_test)
    logger.info(np.array(predictions_pointbypoint).shape)
    plot_results(predictions_pointbypoint, y_test, title = configs.data)
    
    # full-sequence
    # --------------
    prediction_fullseq = model.predict_sequence_full(
        data = x_test,
        window_size = configs['data']['sequence_length'],  # 50
    )
    logger.info(np.array(prediction_fullseq).shape)
    plot_results(prediction_fullseq, y_test, title = configs.data)
    '''
    
if __name__ == "__main__":
    main()
