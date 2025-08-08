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
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
# data pipeline
# from data_provider.todo.data_factory_dl_1 import data_provider
from data_provider.RNNs_type.data_factory import data_provider
# model training
from utils.model_tools import adjust_learning_rate, EarlyStopping
# loss
from utils.ts.losses import mape_loss, mase_loss, smape_loss
# metrics
from utils.ts.metrics_dl import metric, DTW
from utils.plot_results import predict_result_visual
from utils.plot_losses import plot_losses
# log
from utils.model_memory import model_memory_size
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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
        if self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr = self.args.learning_rate,
            )
        elif self.args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr = self.args.learning_rate,
            )
        
        return optimizer
    
    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        model_path.mkdir(parents=True, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = model_path.joinpath("checkpoint.pth")
        
        return model_checkpoint_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path

    def _get_predict_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.predict_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
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
        with open(Path(path).joinpath("result_forecast.txt"), 'a') as file:
            file.write(setting + "  \n")
            file.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, mape accuracy:{mape_accuracy}, mspe:{mspe}, dtw:{dtw}")
            file.write('\n')
            file.write('\n')
            file.close()
        # result2 保存
        np.save(
            Path(path).joinpath('metrics.npy'), 
            np.array([mae, mse, rmse, mape, mape_accuracy, mspe, dtw])
        )
        np.save(Path(path).joinpath('preds.npy'), preds)
        np.save(Path(path).joinpath('trues.npy'), trues)
    
    def _pred_results_save(self, preds, preds_df, path):
        """
        预测结果保存
        """
        if preds is not None:
            np.save(Path(path).joinpath("prediction.npy"), preds) 
        if preds_df is not None:
            preds_df.to_csv(Path(path).joinpath("prediction.csv"), encoding="utf_8_sig", index=False)

    def save_model(self, weights: bool=False):
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

    def load_model(self, weights: str=False):
        """
        模型加载
        """
        logger.info(f'Model Loading model from {self.args.checkpoints}')
        if weights:
            # model weights
            self.model = self.model_dict[self.args.model].Model(self.args)
            self.model.load_state_dict(torch.load(self.args.checkpoints))
        else:
            # whole model
            self.model = torch.load(self.args.checkpoints)
        self.model.eval()

    # TODO
    # def _inverse_data(self, data, outputs, batch_y):
    #     """
    #     输入输出逆转换
    #     """
    #     if data.scale and self.args.inverse:
    #         outputs = data.inverse_target(outputs)
    #         batch_y = data.inverse_target(batch_y)
    #     logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
    #     logger.info(f"debug::batch_y: \n{batch_y} \nbatch_y.shape: {batch_y.shape}")
        
    #     return outputs, batch_y

    def _inverse_data(self, data, outputs, batch_y):
        """
        输入输出逆转换
        """
        if data.scale and self.args.inverse:
            outputs = outputs.numpy()
            batch_y = batch_y.numpy()
            # 数据逆转换 output 最后一个维度转换为与 batch_y 一致: [1, pred_len, enc_in/dec_in]
            if outputs.shape[-1] != batch_y.shape[-1]:
                outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
            # inverse transform
            shape = outputs.shape  # [batch, pred_len, enc_in/dec_in]
            outputs = data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
            batch_y = data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
            # or
            # outputs = data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            # batch_y = data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        # logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
        # logger.info(f"debug::batch_y: \n{batch_y} \nbatch_y.shape: {batch_y.shape}")
        
        return outputs, batch_y

    def train(self, setting):
        """
        模型训练
        """
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
        logger.info(f"Train steps: {train_steps}") 
        # 模型优化器
        optimizer = self._select_optimizer()
        logger.info(f"Train optimizer has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
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
                # logger.info(f"Train step: {i} running...")
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 取出数据
                x_train, y_train = data_batch
                x_train = x_train.float().to(self.device)
                y_train = y_train.float().to(self.device)
                # logger.info(f"debug::x_train: \n{x_train}, \nx_train.shape: {x_train.shape}")
                # logger.info(f"debug::y_train: \n{y_train}, \ny_train.shape: {y_train.shape}")
                
                # 前向传播
                outputs = self.model(x_train)
                # logger.info(f"debug::outputs: \n{outputs}, \noutputs.shape: {outputs.shape}")
                
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
                # logger.info(f"debug::train step: {i}, train loss: {loss}")
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
            vali_loss = self.valid(vali_loader, criterion)
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
        logger.info("Plot and save train/valid losses...")
        plot_losses(
            train_epochs=self.args.train_epochs,
            train_losses=train_losses, 
            valid_losses=vali_losses, 
            label="loss",
            results_path=test_results_path
        )
        # load model
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, vali_loader, criterion):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"Model start validating...")
        # 验证窗口数
        vali_steps = len(vali_loader)
        logger.info(f"Vali total steps: {vali_steps}")
        # 模型验证结果
        vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                # logger.info(f"Vali step: {i} running...")
                x_vali, y_vali = data_batch
                x_vali = x_vali.float().to(self.device)
                y_vali = y_vali.float()
                # 前向传播
                outputs = self.model(x_vali)
                # 计算/保存验证损失
                if self.args.output_size == 1:
                    loss = criterion(outputs.detach().cpu(), y_vali.reshape(-1, 1))
                else:
                    loss = criterion(outputs.detach().cpu(), y_vali)
                vali_loss.append(loss.item())
                # logger.info(f"Valid step: {i}, valid loss: {loss}")
        # 计算验证集上所有 batch 的平均验证损失
        vali_loss = np.average(vali_loss)
        logger.info(f"debug::vali_loss: {vali_loss}")
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"Validating Finished!")
        return vali_loss

    def test_dl1(self, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag="test") 
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
                    predict_result_visual(pred_plot, true_plot, path = Path(test_results_path).joinpath(f'{str(i)}.pdf'))
        # 测试结果保存
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
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
        # TODO test batch_size > 1
        # if test_loader.batch_size > 1:
        #     preds_flat = np.concatenate(preds_flat, axis = 0)
        #     trues_flat = np.concatenate(trues_flat, axis = 0)
        preds_flat = np.concatenate(preds, axis = 0)
        trues_flat = np.concatenate(trues, axis = 0)
        predict_result_visual(preds_flat, trues_flat, path = Path(test_results_path).joinpath("test_pred.png")) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    def test(self, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag="test")
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
                logger.info(f"Test step: {i}")
                x_test, y_test = data_batch
                x_test = x_test.float().to(self.device)
                y_test = y_test.float().to(self.device)
                # 前向传播
                outputs = self.model(x_test)
                outputs = outputs.detach().cpu()
                y_test = y_test.detach().cpu()
                logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
                logger.info(f"debug::y_test: \n{y_test} \ny_test.shape: {y_test.shape}")
                
                # 输入输出逆转换
                # outputs, y_test = self._inverse_data(test_data, outputs, y_test)
                # logger.info(f"debug::outputs: \n{outputs} \noutputs.shape: {outputs.shape}")
                # logger.info(f"debug::y_test: \n{y_test} \ny_test.shape: {y_test.shape}")
                
                # 验证结果收集
                f_dim = -1 if self.args.features == "MS" else 0
                y_pred = outputs[0, :, f_dim:]
                y_test = y_test[0, :, f_dim:]
                # y_pred = test_data.inverse_transform(y_pred.detach().cpu().numpy())
                # y_test = test_data.inverse_transform(y_test.detach().cpu().numpy())
                logger.info(f"debug::y_pred: \n{y_pred} \ny_pred.shape: {y_pred.shape}")
                logger.info(f"debug::y_test: \n{y_test} \ny_test.shape: {y_test.shape}")
                
                preds.append(y_pred)
                trues.append(y_test)
                logger.info(f"debug::preds: \n{preds}")
                logger.info(f"debug::trues: \n{trues}")
                
                # TODO debug
                if i == 1:
                    break
        # 测试结果保存
        logger.info(f"debug::preds: \n{preds}")
        logger.info(f"debug::trues: \n{trues}")
        preds = np.array(preds).reshape(1, -1)
        trues = np.array(trues).reshape(1, -1)
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
        predict_result_visual(preds_flat, trues_flat, path=Path(test_results_path).joinpath("test_pred.png")) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return

    # TODO
    def inspect_model_fit(self, setting, load, train_data, train_loader):
        """
        检验模型拟合情况
        """
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
            logger.info(model_checkpoint_path)
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues= [], []
        # 模型测试
        for x_train, y_train in train_loader:
            x_train = x_train.float().to(self.device)
            y_train = y_train.float().to(self.device)
            # 前向传播
            outputs = self.model(x_train)
            # 验证结果收集
            y_pred = outputs[:, 0, :]
            y_test = y_train[:, 0, :]
            y_pred = train_data.inverse_transform(y_pred.detach().cpu().numpy())
            y_test = train_data.inverse_transform(y_test.detach().cpu().numpy())
            for i in range(len(y_pred)):
                preds.append(y_pred[i][-1])
                trues.append(y_test[i][-1])

    def forecast_dl3(self, rolling_data, scaler, device, show, plt):
        # 预测未知数据的功能
        df = pd.read_csv(self.args.data_path)
        df = pd.concat((df, rolling_data), axis=0).reset_index(drop=True)
        df = df.iloc[:, 1:][-self.args.seq_len:].values  # 转换为nadarry
        pre_data = scaler.transform(df)
        tensor_pred = torch.FloatTensor(pre_data).to(device)
        tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
        model = model
        model.load_state_dict(torch.load('save_model.pth'))
        model.eval()  # 评估模式 
        pred = model(tensor_pred)[0]
    
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        if show:
            # 计算历史数据的长度
            history_length = len(df[:, -1])
            # 为历史数据生成x轴坐标
            history_x = range(history_length)
            plt.figure(figsize=(10, 5))
            # 为预测数据生成x轴坐标
            # 开始于历史数据的最后一个点的x坐标
            prediction_x = range(history_length - 1, history_length + len(pred[:, -1]) - 1)
    
            # 绘制历史数据
            plt.plot(history_x, df[:, -1], label='History')
    
            # 绘制预测数据
            # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
            plt.plot(prediction_x, pred[:, -1], marker='o', label='Prediction')
            plt.axvline(history_length - 1, color='red')  # 在图像的x位置处画一条红色竖线
            # 添加标题和图例
            plt.title("History and Prediction")
            plt.legend()
        return pred

    def rolling_forecast(self, model, device, scaler):
        # 滚动预测
        history_data = pd.read_csv(self.args.data_path)[self.args.target][-self.args.window_size * 4:].reset_index(drop=True)
        pre_data = pd.read_csv(self.args.roolling_data_path)
        columns = pre_data.columns[1:]
        columns = ['forecast' + column for column in columns]
        dict_of_lists = {column: [] for column in columns}
        results = []
        for i in range(int(len(pre_data)/self.args.pre_len)):
            rolling_data = pre_data.iloc[:self.args.pre_len * i]  # 转换为nadarry
            pred = self.predict(model, self.args, device, scaler, rolling_data)
            if self.args.feature == 'MS' or self.args.feature == 'S':
                for i in range(self.args.pred_len):
                    results.append(pred[i][0].detach().cpu().numpy())
            else:
                for j in range(self.args.output_size):
                    for i in range(self.args.pre_len):
                        dict_of_lists[columns[j]].append(pred[i][j])
            print(pred)
        if self.args.feature == 'MS' or self.args.feature == 'S':
            df = pd.DataFrame({'date':pre_data['date'], '{}'.format(self.args.target): pre_data[self.args.target],
                                'forecast{}'.format(self.args.target): pre_data[self.args.target]})
            df.to_csv('Interval-{}'.format(self.args.data_path), index=False)
        else:
            df = pd.DataFrame(dict_of_lists)
            new_df = pd.concat((pre_data,df), axis=1)
            new_df.to_csv('Interval-{}'.format(self.args.data_path), index=False)
        pre_len = len(dict_of_lists['forecast' + self.args.target])
        
        # 绘图
        import matplotlib.pyplot as plt
        plt.figure()
        if self.args.feature == 'MS' or self.args.feature == 'S':
            plt.plot(range(len(history_data)), history_data,label='Past Actual Values')
            plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[self.args.target][:pre_len].tolist(), label='Predicted Actual Values')
            plt.plot(range(len(history_data), len(history_data) + pre_len), results, label='Predicted Future Values')
        else:
            plt.plot(range(len(history_data)), history_data,
                    label='Past Actual Values')
            plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[self.args.target][:pre_len].tolist(), label='Predicted Actual Values')
            plt.plot(range(len(history_data), len(history_data) + pre_len), dict_of_lists['forecast' + self.args.target], label='Predicted Future Values')
        # 添加图例
        plt.legend()
        plt.style.use('ggplot')
        # 添加标题和轴标签
        plt.title('Past vs Predicted Future Values')
        plt.xlabel('Time Point')
        plt.ylabel('Value')
        # 在特定索引位置画一条直线
        plt.axvline(x=len(history_data), color='blue', linestyle='--', linewidth=2)
        # 显示图表
        plt.savefig('forcast.png')
        plt.show()

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




# 测试代码 main 函数
def main():
    pass
 
if __name__ == "__main__":
    main()
