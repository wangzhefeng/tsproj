# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_forecast_dl.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011321
# * Description : description
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
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.model_tools import adjust_learning_rate, EarlyStopping
# metrics
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics_dl import metric, DTW
from utils.polynomial import (
    leg_torch,
    chebyshev_torch, 
    hermite_torch,
    laguerre_torch,
)
from utils.timestamp_utils import from_unix_time
from utils.plot_results import test_result_visual
from utils.plot_losses import plot_losses
# log
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
        # 构建 Transformer 模型
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model].Model(self.args)
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of model parameters: {(total / 1e6):.2f}M')

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
        model_checkpoint_path = f"{model_path}/checkpoint.pth"
        
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
            file.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, mape accuracy:{mape_accuracy}, mspe:{mspe}, dtw: {dtw}")
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
    
    def _model_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder-decoder
        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.amp.autocast("cuda"):
                outputs = _run_model()
        else:
            outputs = _run_model()
        # output
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        return outputs, batch_y
    
    def _fredf_loss(self, batch_index: int, outputs, batch_y, criterion):
        """
        FreDF: Learning to Forecast in the Frequency Domain LOSS
        https://github.com/Master-PLC/FreDF?tab=readme-ov-file#fredf-learning-to-forecast-in-the-frequency-domain

        Args:
            batch_index (_type_): _description_
            outputs (_type_): _description_
            batch_y (_type_): _description_
            criterion (_type_): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        # ------------------------------
        # mask
        # ------------------------------
        if self.args.add_noise and self.args.noise_amp > 0:
            seq_len = self.args.pred_len
            cutoff_freq_percentage = self.args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if self.args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None
        
        # init LOSS
        loss = 0
        # ------------------------------
        # rec lambda
        # ------------------------------
        if self.args.rec_lambda:
            loss_rec = criterion(outputs, batch_y)
            if (batch_index + 1) % 100 == 0:
                print(f"\tloss_rec: {loss_rec.item()}")
            # LOSS
            loss += self.args.rec_lambda * loss_rec 
        # ------------------------------
        # auxi lambda
        # ------------------------------
        if self.args.auxi_lambda:
            # fft shape: [B, P, D]
            if self.args.auxi_mode == "fft":
                loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
            elif self.args.auxi_mode == "rfft":
                if self.args.auxi_type == 'complex':
                    loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                elif self.args.auxi_type == 'complex-phase':
                    loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                elif self.args.auxi_type == 'complex-mag-phase':
                    loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                    loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                elif self.args.auxi_type == 'phase':
                    loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                elif self.args.auxi_type == 'mag':
                    loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                elif self.args.auxi_type == 'mag-phase':
                    loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                    loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                else:
                    raise NotImplementedError
            elif self.args.auxi_mode == "rfft-D":
                loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)
            elif self.args.auxi_mode == "rfft-2D":
                loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)
            elif self.args.auxi_mode == "legendre":
                loss_auxi = leg_torch(outputs, self.args.leg_degree, device=self.device) - leg_torch(batch_y, self.args.leg_degree, device=self.device)
            elif self.args.auxi_mode == "chebyshev":
                loss_auxi = chebyshev_torch(outputs, self.args.leg_degree, device=self.device) - chebyshev_torch(batch_y, self.args.leg_degree, device=self.device)
            elif self.args.auxi_mode == "hermite":
                loss_auxi = hermite_torch(outputs, self.args.leg_degree, device=self.device) - hermite_torch(batch_y, self.args.leg_degree, device=self.device)
            elif self.args.auxi_mode == "laguerre":
                loss_auxi = laguerre_torch(outputs, self.args.leg_degree, device=self.device) - laguerre_torch(batch_y, self.args.leg_degree, device=self.device)
            else:
                raise NotImplementedError
            
            # mask
            if self.mask is not None:
                loss_auxi *= self.mask
            
            # auxi loss
            if self.args.auxi_loss == "MAE":
                # MAE, 最小化 element-wise error 的模长
                loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
            elif self.args.auxi_loss == "MSE":
                # MSE, 最小化 element-wise error 的模长
                loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
            else:
                raise NotImplementedError
            # log
            if (batch_index + 1) % 100 == 0:
                print(f"\tloss_auxi: {loss_auxi.item()}")
            
            # LOSS
            loss += self.args.auxi_lambda * loss_auxi
        
        return loss

    def train(self, setting):
        """
        模型训练
        """
        # 数据集构建
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # checkpoint 保存路径 
        model_checkpoint_path = self._get_model_path(setting)
        logger.info(f"{40 * '-'}")
        logger.info(f"Model checkpoint will be saved in path:")
        logger.info(f"{40 * '-'}")
        logger.info(model_checkpoint_path)
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start training...")
        logger.info(f"{40 * '-'}")
        # time: 模型训练开始时间
        train_start_time = time.time()
        logger.info(f"Train start time: {from_unix_time(train_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        # 训练数据长度
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
        train_result = {}
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 模型训练开始时间
            epoch_start_time = time.time()
            logger.info(f"Epoch: {epoch+1} \tstart time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            # 模型训练
            iter_count = 0
            train_loss = []
            train_result[f"epoch-{epoch+1}"] = {"preds": [], "trues": [], "preds_flat": [], "trues_flat": []}
            # 模型训练模式
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                # logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                # logger.info(f"debug::batch_x: \n{batch_x}, \nbatch_y: \n{batch_y}")
                # logger.info(f"debug::batch_x_mark: \n{batch_x_mark}, \nbatch_y_mark: \n{batch_y_mark}")
                if i == 0:
                    break
                
                if batch_y.shape[1] != (self.args.label_len + self.args.pred_len): 
                    logger.info(f"Train Stop::Data batch_y.shape[1] not equal to (self.args.label_len + self.args.pred_len).")
                    break
                # ------------------------------
                # 前向传播
                # ------------------------------
                outputs, batch_y = self._model_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:].to(self.device)
                # 计算训练损失
                loss = criterion(outputs, batch_y)
                # TODO
                # if not self.args.add_fredf:
                #     loss = criterion(outputs, batch_y)
                # else:
                #     loss = self._fredf_loss(batch_index = i, outputs=outputs, batch_y=batch_y, criterion=criterion)
                # 训练损失收集
                train_loss.append(loss.item())
                train_result[f"epoch-{epoch+1}"]["preds"].append(outputs.detach().cpu().numpy())
                train_result[f"epoch-{epoch+1}"]["trues"].append(batch_y.detach().cpu().numpy())
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 100 == 0:
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
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
            logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")

            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss, vali_preds, vali_trues = self.vali(vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(test_loader, criterion)
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
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
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info(f"{40 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{40 * '-'}")

        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])

        logger.info("Return training results...")
        return self.model, train_result

    def vali(self, vali_loader, criterion):
        """
        模型验证
        """
        # 模型推理模式开启
        self.model.eval()
        # 验证损失收集
        total_loss = []
        # 模型验证结果
        preds, trues = [], []
        # 模型推理
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                # logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                # logger.info(f"debug::batch_x: \n{batch_x}, \nbatch_y: \n{batch_y}")
                # logger.info(f"debug::batch_x_mark: \n{batch_x_mark}, \nbatch_y_mark: \n{batch_y_mark}")
                
                if batch_y.shape[1] != (self.args.label_len + self.args.pred_len):
                    logger.info(f"Valid Stop::Data batch_y.shape[1] not equal to (self.args.label_len + self.args.pred_len).")
                    break
                # logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                # logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                # 前向传播
                outputs, batch_y = self._model_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # 预测值、真实值
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # 计算/保存验证损失
                loss = criterion(pred, true)
                # 验证结果收集
                total_loss.append(loss)
                preds.append(pred)
                trues.append(trues)
        # 计算验证集上所有 batch 的平均验证损失
        total_loss = np.average(total_loss)
        # 计算模型输出
        self.model.train()
        
        return total_loss, preds, trues

    def test(self, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag='test')
        # 模型加载
        if load: 
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            logger.info(model_checkpoint_path)
        # 测试结果保存地址 
        test_results_path = self._get_test_results_path(setting)
        logger.info(f"{40 * '-'}")
        logger.info(f"Test results will be saved in path:")
        logger.info(f"{40 * '-'}")
        logger.info(test_results_path)
        # 模型开始测试
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start testing...")
        logger.info(f"{40 * '-'}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        trues, preds = [], []
        trues_flat, preds_flat = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                logger.info(f"test step: {i}")
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                # logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                if batch_y.shape[1] != (self.args.label_len + self.args.pred_len):
                    logger.info(f"Test Stop::Data batch_y.shape[1] not equal to (self.args.label_len + self.args.pred_len).")
                    break
                # 前向传播
                outputs, batch_y = self._model_forward(
                    batch_x = batch_x, 
                    batch_y = batch_y, 
                    batch_x_mark = batch_x_mark, 
                    batch_y_mark = batch_y_mark,
                )
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # 输入输出逆转换
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape  # [batch, pred_len, 7]
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape) 
                # logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
                # logger.info(f"debug::batch_y: \n{batch_y} \nbatch_y.shape: {batch_y.shape}")
                
                # 预测值/真实值提取
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]
                # logger.info(f"debug::pred: \n{pred} \npred shape: {pred.shape}")
                # logger.info(f"debug::true: \n{true} \ntrue shape: {true.shape}")
                
                # 预测结果收集
                preds.append(pred)
                trues.append(true) 
                for batch_idx in range(self.args.batch_size):
                    preds_flat.append(pred[batch_idx, :, -1].tolist())
                    trues_flat.append(true[batch_idx, :, -1].tolist())
                
                # 预测数据可视化
                if i % 20 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    true_plot = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    test_result_visual(pred_plot, true_plot, path = os.path.join(test_results_path, str(i) + '.pdf'))
        # 预测/实际标签处理
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # logger.info(f"Test results: preds: \n{preds} \npreds.shape: {preds.shape}")
        # logger.info(f"Test results: trues: \n{trues} \ntrues.shape: {trues.shape}")

        logger.info(f"{40 * '-'}")
        logger.info(f"Test result saving...")
        logger.info(f"{40 * '-'}")
        # 测试结果保存
        self._test_results_save(preds, trues, setting, path = test_results_path)
        logger.info(f"Test metric results have been saved in: {test_results_path}")
        
        # 测试结果可视化
        preds_flat = np.concatenate(preds_flat, axis = 0)
        trues_flat = np.concatenate(trues_flat, axis = 0) 
        test_result_visual(preds_flat, trues_flat, path = os.path.join(test_results_path, "test_prediction.png"))
        # logger.info(f"Test results: preds_flat: {preds_flat} \npreds_flat.shape: {preds_flat.shape}")
        # logger.info(f"Test results: trues_flat: {trues_flat} \ntrues_flat.shape: {trues_flat.shape}")
        logger.info(f"Test visual results saved in {test_results_path}")

        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    def forecast(self, setting, load: bool=True):
        """
        模型预测
        """
        # 构建预测数据集
        pred_data, pred_loader = self._get_data(flag='pred')
        
        # 模型加载
        if load:
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            logger.info(model_checkpoint_path)
        
        # 模型预测结果保存地址
        pred_results_path = self._get_predict_results_path(setting)
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results will be saved in:")
        logger.info(f"{40 * '-'}")
        logger.info(pred_results_path)
        
        # 模型预测
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start forecasting...")
        logger.info(f"{40 * '-'}")
        # 模型评估模式
        self.model.eval()
        # 模型预测
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                logger.info(f"forecast step: {i}")
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                # logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")

                # 前向传播
                outputs, batch_y = self._model_forward(
                    batch_x = batch_x,
                    batch_y = batch_y,
                    batch_x_mark = batch_x_mark,
                    batch_y_mark = batch_y_mark,
                )
                outputs = outputs.detach().cpu().numpy()
                # logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
                # logger.info(f"debug::batch_y: \n{batch_y} \nbatch_y.shape: {batch_y.shape}")

                # 预测值逆转换
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape  # [batch, pred_len, 7]
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    # outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    outputs = pred_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                # 预测值提取
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, :, f_dim:]
                # logger.info(f"pred: \n{pred} \npred shape: {pred.shape}")

                # 预测值收集
                preds.append(pred)

                # 预测数据可视化
                inputs = batch_x.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = inputs.shape
                    inputs = pred_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                true_plot = inputs[0, :, -1]
                test_result_visual(pred_plot, true_plot, path = os.path.join(pred_results_path, 'forecasting_prediction.png'))
        
        # 最终预测值
        preds = np.array(preds).squeeze()
        logger.info(f"preds: \n{preds} \npreds shape: {preds.shape}")
        preds_df = pd.DataFrame({
            "timestamp": pd.date_range(pred_data.forecast_start_time, periods=self.args.pred_len, freq=self.args.freq),
            "predict_value": preds,
        })
        logger.info(f"preds_df: \n{preds_df} \npreds_df.shape: {preds_df.shape}")
        
        # 最终预测值保存
        np.save(os.path.join(pred_results_path, "prediction.npy"), preds) 
        preds_df.to_csv(
            os.path.join(pred_results_path, "prediction.csv"), 
            encoding="utf_8_sig", 
            index=False
        )
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results have been saved in:")
        logger.info(f"{40 * '-'}")
        logger.info(pred_results_path)

        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecasting Finished!")
        logger.info(f"{40 * '-'}")

        return preds, preds_df

    def train_v2(self, setting, ii):
        """
        模型训练
        """
        # ------------------------------
        # 数据集构建
        # ------------------------------
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        # ------------------------------
        # checkpoint 保存路径
        # ------------------------------
        model_checkpoint_path = self._get_model_path(setting)
        # ------------------------------
        # 测试结果保存地址
        # ------------------------------
        test_results_path = self._get_test_results_path(setting)
        # ------------------------------
        # 模型训练
        # ------------------------------
        # time: 模型训练开始时间
        time_now = time.time()
        # 训练数据长度
        train_steps = len(train_loader)
        logger.info(f"train_steps: {train_steps}")
        # 模型优化器
        optimizer = self._select_optimizer()
        # 模型损失函数
        criterion = self._select_criterion()
        # 早停类实例
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        # 训练、验证结果收集
        # train_result = {}
        train_losses, val_losses = [], []
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 模型训练开始时间
            epoch_time = time.time()
            # 模型训练
            iter_count = 0
            train_loss = []
            self.model.train()
            # 训练、验证结果收集
            # train_result[f"{epoch}"] = {"preds": [], "trues": [], "preds_flat": [], "trues_flat": []}
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_x.shape[1] - (self.args.label_len + self.args.pred_len) != batch_y.shape[1]:
                    break
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder-decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                """
                # TODO ------------------------------
                # TODO ------------------------------
                # pred/true label
                outputs_1 = outputs[:, -self.args.pred_len:, :]  # 预测 label
                outputs_1 = outputs_1.detach().cpu().numpy()
                batch_y_1 = batch_y[:, -self.args.pred_len:, :].to(self.device)  # 实际 label
                batch_y_1 = batch_y_1.detach().cpu().numpy()
                # 输入输出逆转换
                if vali_data.scale and self.args.inverse:
                    if outputs_1.shape[-1] != batch_y_1.shape[-1]:
                        outputs_1 = np.tile(outputs_1, [1, 1, int(batch_y_1.shape[-1] / outputs_1.shape[-1])])
                    outputs_1 = vali_data.inverse_transform(outputs_1.reshape(batch_y_1.shape[0] * batch_y_1.shape[1], -1)).reshape(batch_y_1.shape)
                    batch_y_1 = vali_data.inverse_transform(batch_y_1.reshape(batch_y_1.shape[0] * batch_y_1.shape[1], -1)).reshape(batch_y_1.shape)
                # 输出取 target
                f_dim = -1 if self.args.features == 'MS' else 0  # 目标特征维度
                pred = outputs_1[:, :, f_dim:]
                true = batch_y_1[:, :, f_dim:]
                # logger.info(f"pred: {pred}")
                # logger.info(f"true: {true}")
                # 预测结果收集
                train_result[f"{epoch}"]["preds"].append(pred)
                train_result[f"{epoch}"]["trues"].append(true)
                train_result[f"{epoch}"]["preds_flat"].append(pred[0, :, -1].tolist())
                train_result[f"{epoch}"]["trues_flat"].append(true[0, :, -1].tolist())
                # TODO ------------------------------
                # TODO ------------------------------
                """
                # pred/true label
                f_dim = -1 if self.args.features == 'MS' else 0  # 目标特征维度
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # update/save loss
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                # 日志打印：当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 100 == 0:
                    logger.info(f"Epoch: {epoch + 1}, \tIters: {i + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tSpeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                # ------------------------------
                # 后向传播、参数优化更新
                # ------------------------------
                loss.backward()
                optimizer.step()
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            logger.info(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time}")
            # 日志打印：训练 epoch、每个 epoch 训练后的 train_loss、vali_loss、test_loss
            train_loss = np.average(train_loss)
            vali_loss, preds_flat_vali, trues_flat_vali = self.vali_v2(vali_data, vali_loader, criterion, setting, ii, epoch)
            # test_loss, preds_flat_test, trues_flat_test = self.vali_v2(test_data, test_loader, criterion, setting, ii, epoch)
            train_losses.append(train_loss)
            val_losses.append(vali_loss)
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")#, Test Loss: {test_loss:.7f}")
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
        """
        # TODO ------------------------------
        # TODO ------------------------------
        # 训练结果
        # logger.info(f"train_result: {train_result}")
        for epoch_idx in range(self.args.train_epochs):
            # 预测/实际标签处理
            preds = np.concatenate(train_result[f"{epoch}"]["preds"], axis = 0)
            trues = np.concatenate(train_result[f"{epoch}"]["trues"], axis = 0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            # 预测结果可视化
            preds_flat = np.concatenate(preds, axis = 0)
            trues_flat = np.concatenate(trues, axis = 0)
            predict_results_path = self._get_predict_results_path(setting)
            test_result_visual(preds_flat, trues_flat, path = os.path.join(predict_results_path, f"load_prediction-train-{ii}-{epoch_idx}.png"))
        # TODO ------------------------------
        # TODO ------------------------------
        """
        # plot losses
        plot_losses(
            train_epochs=self.args.train_epochs, 
            train_losses=train_losses, 
            val_losses=val_losses, 
            label="loss",
            results_path=test_results_path
        )
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
        
        return self.model

    def vali_v2(self, vali_data, vali_loader, criterion, setting, ii, epoch):
        """
        模型验证
        """
        # 模型推理模式开启
        self.model.eval()
        # 验证损失收集
        total_loss = []
        # 模型验证结果
        trues, preds = [], []
        trues_flat, preds_flat = [], []
        # 模型推理
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # ------------------------------
                # 数据预处理
                # ------------------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_x.shape[1] - (self.args.label_len + self.args.pred_len) != batch_y.shape[1]:
                    break
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                """
                # TODO ------------------------------
                # TODO ------------------------------
                # pred/true label
                outputs_1 = outputs[:, -self.args.pred_len:, :]  # 预测 label
                outputs_1 = outputs_1.detach().cpu().numpy()
                batch_y_1 = batch_y[:, -self.args.pred_len:, :].to(self.device)  # 实际 label
                batch_y_1 = batch_y_1.detach().cpu().numpy()
                # 输入输出逆转换
                if vali_data.scale and self.args.inverse:
                    if outputs_1.shape[-1] != batch_y_1.shape[-1]:
                        outputs_1 = np.tile(outputs_1, [1, 1, int(batch_y_1.shape[-1] / outputs_1.shape[-1])])
                    outputs_1 = vali_data.inverse_transform(outputs_1.reshape(batch_y_1.shape[0] * batch_y_1.shape[1], -1)).reshape(batch_y_1.shape)
                    batch_y_1 = vali_data.inverse_transform(batch_y_1.reshape(batch_y_1.shape[0] * batch_y_1.shape[1], -1)).reshape(batch_y_1.shape)
                # 输出取 target
                f_dim = -1 if self.args.features == 'MS' else 0  # 目标特征维度
                pred = outputs_1[:, :, f_dim:]
                true = batch_y_1[:, :, f_dim:]
                # 预测结果收集
                preds.append(pred)
                trues.append(true)
                preds_flat.append(pred[0, :, -1].tolist())
                trues_flat.append(true[0, :, -1].tolist())
                # TODO ------------------------------
                # TODO ------------------------------
                """
                # pred/true label
                f_dim = -1 if self.args.features == 'MS' else 0  # 目标特征维度
                outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 预测 label
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 实际 label
                # 计算/保存验证损失
                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                # ------------------------------
                # 验证损失收集
                # ------------------------------
                total_loss.append(loss)
        """
        # TODO ------------------------------
        # TODO ------------------------------
        # 预测/实际标签处理
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # 预测结果可视化
        preds_flat = np.concatenate(preds, axis = 0)
        trues_flat = np.concatenate(trues, axis = 0)
        logger.info(f"preds_flat.shape: {len(preds_flat)}")
        logger.info(f"trues_flat.shape: {len(trues_flat)}")
        predict_results_path = self._get_predict_results_path(setting)
        test_result_visual(preds_flat, trues_flat, path = os.path.join(predict_results_path, f"load_prediction-vali-{ii}-{epoch}.png"))
        # TODO ------------------------------
        # TODO ------------------------------
        """
        # 计算所有 batch 的平均验证损失
        total_loss = np.average(total_loss)
        # 计算模型输出
        self.model.train()
        
        return total_loss, preds_flat, trues_flat

    def test_v2(self, setting, load = True):
        """
        模型测试
        """
        # ------------------------------
        # 数据集构建
        # ------------------------------
        test_data, test_loader = self._get_data(flag='test')
        # ------------------------------
        # 模型加载
        # ------------------------------
        if load:
            logger.info("Loading best model...")
            best_model_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(best_model_path)["model"])
        # ------------------------------
        # 测试结果保存地址
        # ------------------------------
        test_results_path = self._get_test_results_path(setting)
        predict_results_path = self._get_predict_results_path(setting)
        # ------------------------------
        # 模型测试
        # ------------------------------
        # 模型推理模式开启
        self.model.eval()
        # 测试结果收集
        trues, preds = [], []
        trues_flat, preds_flat = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                logger.info(f"test step: {i}")
                logger.info("-" * 15)
                # ------------------------------
                # 数据预处理
                # ------------------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
            
                if batch_x.shape[1] - (self.args.label_len + self.args.pred_len) != batch_y.shape[1]:
                    break
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                
                # encoder-decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # pred/true label
                outputs = outputs[:, -self.args.pred_len:, :]  # 预测 label
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # 实际 label
                batch_y = batch_y.detach().cpu().numpy()
                # 输入输出逆转换
                if test_data.scale and self.args.inverse:
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    # outputs = test_data.inverse_transform(outputs)
                    # batch_y = test_data.inverse_transform(batch_y)
                    outputs = test_data.inverse_transform(outputs.reshape(batch_y.shape[0] * batch_y.shape[1], -1)).reshape(batch_y.shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(batch_y.shape[0] * batch_y.shape[1], -1)).reshape(batch_y.shape)
                # 输出取 target
                f_dim = -1 if self.args.features == 'MS' else 0  # 目标特征维度
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]
                # logger.info(f"pred: {pred}")
                # logger.info(f"true: {true}")
                # 预测数据可视化
                if i % 20 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        inputs = test_data.inverse_transform(inputs.reshape(inputs.shape[0] * inputs.shape[1], -1)).reshape(inputs.shape)
                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    test_result_visual(pd, gt, path = os.path.join(test_results_path, str(i) + '.pdf'))
                # 预测结果收集
                preds.append(pred)
                trues.append(true)
                preds_flat.append(pred[0, :, -1].tolist())
                trues_flat.append(true[0, :, -1].tolist())
        # ------------------------------
        # 预测/实际标签处理
        # ------------------------------
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # ------------------------------
        # 预测结果可视化
        # ------------------------------
        preds_flat = np.concatenate(preds, axis = 0)
        trues_flat = np.concatenate(trues, axis = 0)
        logger.info(f"preds_flat.shape: {len(preds_flat)}")
        logger.info(f"trues_flat.shape: {len(trues_flat)}")
        test_result_visual(preds_flat, trues_flat, path = os.path.join(predict_results_path, "load_prediction.png"))
        # ------------------------------
        # 结果保存
        # ------------------------------
        self._test_results_save(preds, trues, setting, test_results_path)

        return




def main():
    pass

if __name__ == '__main__':
    main()
