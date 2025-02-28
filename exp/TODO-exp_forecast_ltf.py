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
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider_new
from utils.model_tools import adjust_learning_rate, EarlyStopping
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics_dl import metric, DTW
from utils.visual import test_result_visual
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Forecast(Exp_Basic):
    """
    搭建模型 Transoformer 模型
    """

    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 构建 Transformer 模型
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of parameters: {(total / 1e6):.2f}M')

        return model

    def _get_data(self, flag: str):
        """
        数据集构建

        Args:
            flag (str): 任务类型, ["train", "val", "test"]

        Returns:
            _type_: Dataset, DataLoader
        """
        data_set, data_loader = data_provider_new(self.args, flag)
        
        return data_set, data_loader

    def _select_criterion(self, loss_name: str = "MSE"):
        """
        评价指标
        """
        if loss_name == "MSE":
            return nn.MSELoss()
        elif loss_name == "MAPE":
            return mape_loss()
        elif loss_name == "MASE":
            return mase_loss()
        elif loss_name == "SMAPE":
            return smape_loss()

    def _select_optimizer(self):
        """
        优化器
        """
        model_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.learning_rate
        )
        
        return model_optim
    
    # TODO
    def _get_path(self, setting):
        """
        模型及其结果保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        best_model_path = f"{model_path}/checkpoint.pth"

        # 测试结果保存路径 
        test_results_path = os.path.join(self.args.test_results, setting + "/")
        os.makedirs(test_results_path, exist_ok=True)

        # 预测结果保存路径
        preds_results_path = os.path.join(self.args.predict_results, setting + "/")
        os.makedirs(preds_results_path, exist_ok=True)

        return best_model_path, test_results_path, preds_results_path

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        best_model_path = f"{model_path}/checkpoint.pth"
        
        return best_model_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting + "/")
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _get_predict_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.predict_results, setting + "/")
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _test_results_save(self, preds, trues, setting, test_results_path):
        """
        测试结果保存

        Args:
            preds (_type_): _description_
            trues (_type_): _description_
            setting (_type_): _description_
            test_results_path (_type_): _description_
        """
        # 计算测试结果评价指标
        mse, rmse, mae, mape, accuracy, mspe = metric(preds, trues)
        dtw = DTW(preds, trues) if self.args.use_dtw else -999
        logger.info(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, accuracy:{accuracy}, mspe:{mspe}")
        # result1 保存
        f = open(os.path.join(test_results_path, "result_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, accuracy:{accuracy}, mspe:{mspe}")
        f.write('\n')
        f.write('\n')
        f.close()
        # result2 保存
        np.save(test_results_path + 'metrics.npy', np.array([mae, mse, rmse, mape, accuracy, mspe, dtw]))
        np.save(test_results_path + 'preds.npy', preds)
        np.save(test_results_path + 'trues.npy', trues)

    def train(self, setting, ii):
        """
        模型训练
        """
        # ------------------------------
        # 数据集构建
        # ------------------------------
        train_data, train_loader = self._get_data(flag='train', pre_data=None)
        vali_data, vali_loader = self._get_data(flag='val', pre_data=None)
        # test_data, test_loader = self._get_data(flag='test', pre_data=None)
        # ------------------------------
        # checkpoint 保存路径
        # ------------------------------
        best_model_path = self._get_model_path(setting)
        # ------------------------------
        # 模型训练
        # ------------------------------
        # time: 模型训练开始时间
        time_now = time.time()
        # 训练数据长度
        train_steps = len(train_loader)
        logger.info(f"train_steps: {train_steps}")
        # 模型优化器
        model_optim = self._select_optimizer()
        # 模型损失函数
        criterion = self._select_criterion(self.args.loss)
        # 早停类实例
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        # 训练、验证结果收集
        # train_result = {}
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
                model_optim.zero_grad()
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
                model_optim.step()
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            logger.info(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time}")
            # 日志打印：训练 epoch、每个 epoch 训练后的 train_loss、vali_loss、test_loss
            train_loss = np.average(train_loss)
            vali_loss, preds_flat_vali, trues_flat_vali = self.vali(vali_data, vali_loader, criterion, setting, ii, epoch)
            # test_loss = self.vali(test_loader, criterion)
            logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")#, Test Loss: {test_loss:.7f}")
            # 早停机制、模型保存
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, Early stopping...")
                break
            # 学习率调整
            adjust_learning_rate(model_optim, epoch + 1, self.args)
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
            test_result_visual(trues_flat, preds_flat, path = os.path.join(predict_results_path, f"load_prediction-train-{ii}-{epoch_idx}.png"))
        # TODO ------------------------------
        # TODO ------------------------------
        """
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def vali(self, vali_data, vali_loader, criterion, setting, ii, epoch):
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
        test_result_visual(trues_flat, preds_flat, path = os.path.join(predict_results_path, f"load_prediction-vali-{ii}-{epoch}.png"))
        # TODO ------------------------------
        # TODO ------------------------------
        """
        # 计算所有 batch 的平均验证损失
        total_loss = np.average(total_loss)
        # 计算模型输出
        self.model.train()
        
        return total_loss, preds_flat, trues_flat

    def test(self, setting):
        """
        模型测试
        """
        # ------------------------------
        # 数据集构建
        # ------------------------------
        test_data, test_loader = self._get_data(flag='test', pre_data=None)
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        best_model_path = self._get_model_path(setting)
        self.model.load_state_dict(torch.load(best_model_path))
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
                    test_result_visual(gt, pd, path = os.path.join(test_results_path, str(i) + '.pdf'))
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
        test_result_visual(trues_flat, preds_flat, path = os.path.join(predict_results_path, "load_prediction.png"))
        # ------------------------------
        # 结果保存
        # ------------------------------
        self._test_results_save(preds, trues, setting, test_results_path)

        return

    # TODO
    def predict_rolling(self, setting, load = False):
        # ------------------------------
        # 滑动窗口预测，读取 pre_data
        # ------------------------------
        pre_data = pd.read_csv(self.args.root_path + self.args.rolling_data_path) if self.args.rollingforecast else None
        # ------------------------------
        # 加载模型
        # ------------------------------
        if load:
            logger.info("Loading best model...")
            best_model_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device("cpu")))
        # ------------------------------
        # 预测模型结果保存路径
        # ------------------------------
        predict_results_path = self._get_predict_results_path(setting)
        # ------------------------------
        # 评估模式
        # ------------------------------
        self.model.eval()
        # ------------------------------
        # 模型预测
        # ------------------------------
        final_trues = []
        final_preds = []
        for i in [0] if pre_data is None else range(int(len(pre_data) / self.args.pred_len) - 1):
            # 数据集构建
            if i == 0:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=None)
                logger.info(f'滚动预测第 {i + 1} 次')
            else:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[: i * self.args.pred_len])
                logger.info(f'滚动预测第 {i + 1} 次')
            # 进行滚动预测
            with torch.no_grad():
                for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    # 数据预处理
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    logger.info(f"batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape}")
                    logger.info(f"batch_x_mark.shape: {batch_x_mark.shape}, batch_y_mark.shape: {batch_y_mark.shape}")
                    # decoder input
                    if self.args.padding == 0:
                        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(batch_y.device)
                    elif self.args.padding == 1:
                        dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(batch_y.device)
                    # logger.info(f"dec_inp.shape: {dec_inp.shape}")
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # logger.info(f"dec_inp.shape: {dec_inp.shape}")
                    
                    # encoder-decoder
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    logger.info(f"1.outputs.shape: {outputs.shape}")
                    
                    if pred_data.scale and pred_data.inverse:
                        shape = outputs.shape
                        outputs = pred_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    # 预测值逆转换
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    logger.info(f"2.outputs.shape: {outputs.shape}")
                    
                    # 预测值收集
                    if self.args.features == 'MS':
                        for i in range(self.args.pred_len):
                            pred = outputs[0][i][outputs.shape[2] - 1]  # 取最后一个预测值即对应target列
                            final_preds.append(pred)
                    else:
                        # TODO
                        for i in range(self.args.pred_len):
                            final_preds.append(outputs[0][i])
                            
        logger.info(f"3.final_preds: {len(final_preds)}")
        # ------------------------------
        # 保存结果
        # ------------------------------
        if self.args.rollingforecast:
            df = pd.DataFrame({
                'true': pre_data[self.args.target][:len(final_preds)],
                'pred': final_preds,
            })
        else:
            df = pd.DataFrame({
                'true':  None,
                'pred': final_preds,
            })
        logger.info(f"df: \n{df}")
        logger.info(f"df.shape: \n{df.shape}")
        df.to_csv(os.path.join(predict_results_path, f"{self.args.target}-ForecastResults.csv"), index=False)
        # ------------------------------
        # 结果可视化
        # ------------------------------
        if self.args.show_results:
            test_result_visual(df, os.path.join(predict_results_path, f"{self.args.target}-ForecastResults.png"))

    # TODO
    def predict(self, setting, load = False):
        # ------------------------------
        # 构建预测数据集
        # ------------------------------
        pred_data, pred_loader = self._get_data(flag='pred', pre_data=None)
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        best_model_path = self._get_model_path(setting)
        self.model.load_state_dict(torch.load(best_model_path))
        # ------------------------------
        # 评估模式
        # ------------------------------
        self.model.eval()
        # ------------------------------
        # 模型预测
        # ------------------------------
        preds = []
        # 进行滚动预测
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                logger.info(f"batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape}")
                logger.info(f"batch_x_mark.shape: {batch_x_mark.shape}, batch_y_mark.shape: {batch_y_mark.shape}")
                
                # logger.info(f"batch_y.shape: {batch_y.shape}")
                # logger.info(f"batch_y[:, -self.args.pred_len:, :].shape: {batch_y[:, -self.args.pred_len:, :].shape}")
                # logger.info(f"batch_y.shape[0]: {batch_y.shape[0]}")
                # logger.info(f"self.args.pred_len: {self.args.pred_len}")
                # logger.info(f"batch_y.shape[-1]: {batch_y.shape[-1]}")
                
                # TODO decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # logger.info(f"dec_inp.shape: {dec_inp.shape}")
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                # logger.info(f"dec_inp.shape: {dec_inp.shape}")
                
                if self.args.padding == 0:
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(batch_y.device)
                elif self.args.padding == 1:
                    dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(batch_y.device)
                logger.info(f"dec_inp.shape: {dec_inp.shape}")
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                logger.info(f"dec_inp.shape: {dec_inp.shape}")
                # encoder-decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # 预测值逆转换
                if pred_data.scale and self.args.inverse:
                    outputs = pred_data.inverse_transform(outputs)
                # 预测输出值
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
            
                # 预测值收集
                preds.append(outputs.detach().cpu().numpy())
            # 最终预测值
            preds = np.array(preds)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]).squeeze()
            logger.info(f"preds: \n{preds}")
            logger.info(f"preds.shape: {preds.shape}")
            
            # 最终预测值保存
            predict_results_path = self._get_predict_results_path(setting)
            np.save(predict_results_path + 'prediction.npy', preds)




def main():
    pass

if __name__ == '__main__':
    main()
