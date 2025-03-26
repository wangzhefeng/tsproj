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

from data_provider.data_factory import data_provider_new
from exp.exp_basic import Exp_Basic
from utils.model_tools import adjust_learning_rate, EarlyStopping
# metrics
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics_dl import metric, DTW
# log
from utils.log_util import logger
from utils.timestamp_utils import from_unix_time

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 构建 Transformer 模型
        model = self.model_dict[self.args.model].Model(self.args).float()
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
        """
        data_set, data_loader = data_provider_new(self.args, flag)
        
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
    
    # TODO 
    def _train_model_save(self, epoch, model, optimizer=None, scheduler=None, model_path=None):
        """
        模型保存
        """
        training_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
    
    # TODO
    def _train_model_load(self, model, optmizer=None, scheduler=None, model_path=None):
        """
        模型加载
        """
        checkpoints = torch.load(model_path)
        epoch = checkpoints["epoch"]
        if model:
            model.load_state_dict(checkpoints["model"])
        if optmizer:
            optmizer.load_state_dict(checkpoints["optmizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoints["scheduler"])

        return {
            "epoch": epoch, 
            "model": model, 
            "optimizer": optmizer, 
            "scheduler": scheduler
        }
    
    # TODO
    def train_recover(self, setting, log_dir):
        # 模型加载
        if os.path.exists(log_dir):
            load_input = self._train_model_load()
            start_epoch = load_input["epoch"]
            logger.info(f"加载 Epoch: {load_input['epoch']} 成功.")
        else:
            start_epoch = 0
            logger.info(f"无保存模型，将从头开始训练...")
        
        for epoch in range(start_epoch+1, self.args.train_epochs):
            self.train(setting)
    
    def _test_result_visual(self, preds, trues, path='./pic/test.pdf'):
        """
        Results visualization
        """
        # 设置绘图风格
        # plt.style.use('ggplot')
        # 画布
        fig = plt.figure(figsize = (25, 5))
        # 创建折线图
        plt.plot(trues, lw=1, label='Trues')
        plt.plot(preds, lw=1, ls="--", label='Preds')
        # 增强视觉效果
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        # plt.ylim(5, 20)
        plt.title('实际值 vs 预测值')
        plt.grid(True)
        plt.savefig(path, bbox_inches='tight')
        plt.show();
    
    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
        # 计算测试结果评价指标
        mse, rmse, mae, mape, accuracy, mspe = metric(preds, trues)
        dtw = DTW(preds, trues) if self.args.use_dtw else -999
        logger.info(f"Test results: mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, accuracy:{accuracy}, mspe:{mspe}")
        # result1 保存
        f = open(os.path.join(path, "result_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, accuracy:{accuracy}, mspe:{mspe}")
        f.write('\n')
        f.write('\n')
        f.close()
        # result2 保存
        np.save(os.path.join(path, 'metrics.npy'), np.array([mae, mse, rmse, mape, accuracy, mspe, dtw]))
        np.save(os.path.join(path, 'preds.npy'), preds)
        np.save(os.path.join(path, 'trues.npy'), trues)

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # logger.info(f"debug::dec_inp.shape: {dec_inp.shape}")
        
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
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
    
    def _predict_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # logger.info(f"debug::dec_inp.shape: {dec_inp.shape}")
        
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

    def train(self, setting):
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
        logger.info(f"Train model will be saved in path: {model_checkpoint_path}.")
        # ------------------------------
        # 模型训练
        # ------------------------------
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
        # 自动混合精度训练
        if self.args.use_amp:
            logger.info(f"debug::self.args.use_amp: {self.args.use_amp}")
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
                # ------------------------------
                # 前向传播
                # ------------------------------
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 计算训练损失
                loss = criterion(outputs, batch_y)
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
            # test_loss, test_preds, test_trues = self.vali(test_loader, criterion)
            # logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
            logger.info(f"Epoch: {epoch + 1}, \tSteps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            
            # 早停机制、模型保存
            early_stopping(vali_loss, epoch, self.model, optimizer, model_checkpoint_path)
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, \tEarly stopping...")
                break

            # 学习率调整
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        # ------------------------------
        # 模型加载
        # ------------------------------
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
        
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
                # 前向传播
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
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

    def test(self, setting, test = 0):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag='test')
        # 模型加载
        if test:
            logger.info("Loading pretrained model...")
            model_checkpoint_path = self._get_model_path(setting)
            # self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
            self.model.load_state_dict(torch.load(model_checkpoint_path))
            logger.info(f"Pretrained model has loaded from {model_checkpoint_path}")
        # 测试结果保存地址
        test_results_path = self._get_test_results_path(setting)
        logger.info(f"Test results will be saved in path: {test_results_path}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        trues, preds = [], []
        trues_flat, preds_flat = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                logger.info(f"test step: {i}")
                logger.info("-" * 15)
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                # 前向传播
                outputs, batch_y = self._predict_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                logger.info(f"debug::outputs.shape: {outputs.shape}")
                logger.info(f"debug::batch_y.shape: {batch_y.shape}")
                # 输入输出逆转换
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape  # [batch, pred_len, 7]
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # outputs = test_data.inverse_transform(outputs.unsqueeze()).reshape(shape)
                    # batch_y = test_data.inverse_transform(batch_y.unsqueeze()).reshape(shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                logger.info(f"debug::outputs.shape: {outputs.shape}")
                logger.info(f"debug::batch_y.shape: {batch_y.shape}")
                # 预测值/真实值
                pred = outputs
                true = batch_y 
                logger.info(f"debug::pred: \n{pred}")
                logger.info(f"debug::true: \n{true}")
                # 预测结果收集 
                preds.append(pred)
                trues.append(true)
                preds_flat.append(pred[0, :, -1].tolist())
                trues_flat.append(true[0, :, -1].tolist())
                # 预测数据可视化
                if i % 20 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    self._test_result_visual(pd, gt, path = os.path.join(test_results_path, str(i) + '.pdf')) 
                if i == 1:
                    break
        # 预测/实际标签处理
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        logger.info(f"Test results: preds: \n{preds} \npreds.shape: {pred.shape}")
        logger.info(f"Test results: trues: \n{trues} \ntrues.shape: {trues.shape}")
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logger.info(f"Test results: preds.shape: {pred.shape}, trues.shape: {trues.shape}")
        
        # 测试结果保存
        self._test_results_save(preds, trues, setting, path = test_results_path)
        logger.info(f"Test results saved in {test_results_path}")
        
        # 测试结果可视化
        trues_flat = np.concatenate(trues, axis = 0)
        preds_flat = np.concatenate(preds, axis = 0)
        logger.info(f"Test results: trues_flat length: {len(trues_flat)} preds_flat length: {len(preds_flat)}")
        self._test_result_visual(preds_flat, trues_flat, path = os.path.join(test_results_path, "test_prediction.png"))
        logger.info(f"Test results visual saved in {test_results_path}")

        return
 
    def forecast(self, setting, load = False):
        """
        模型预测
        """
        # 构建预测数据集
        pred_data, pred_loader = self._get_data(flag='pred')
        # 模型加载
        if load:
            logger.info("Loading pretrained model...")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path))
            logger.info(f"Pretrained model has loaded from {model_checkpoint_path}.")
        # 模型预测结果保存地址
        predict_results_path = self._get_predict_results_path(setting)
        logger.info(f"Forecast results will be saved in path: {predict_results_path}")
        # 模型评估模式
        self.model.eval()
        # 模型预测
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                logger.info(f"forecast step: {i}")
                logger.info("-" * 15)
                # 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                logger.info(f"debug::batch_x.shape: {batch_x.shape} batch_y.shape: {batch_y.shape}")
                logger.info(f"debug::batch_x_mark.shape: {batch_x_mark.shape} batch_y_mark.shape: {batch_y_mark.shape}")
                # 前向传播
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                logger.info(f"debug::outputs: {outputs} \noutputs.shape: {outputs.shape}")
                # 预测值逆转换
                if pred_data.scale and self.args.inverse:
                    outputs = pred_data.inverse_transform(outputs)
                logger.info(f"debug::outputs: {outputs} \noutputs.shape: {outputs.shape}")
                # 预测值
                pred = outputs
                logger.info(f"debug::pred: {pred} \npred.shape: {pred.shape}")
                # 预测值收集
                preds.append(pred)
        # 最终预测值
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])#.squeeze()
        # preds_df = pd.DataFrame({
        #     "timestamp": pd.date_range(self.args.pred_start_time, self.args.pred_end_time, freq=self.args.freq),
        #     "predict_value": preds, 
        # })
        logger.info(f"preds: \n{preds} \npreds.shape: {preds.shape}")
        # logger.info(f"preds_df: \n{preds_df} \npreds_df.shape: {preds_df.shape}")
        
        # 最终预测值保存
        np.save(os.path.join(predict_results_path, "prediction.npy"), preds) 
        # preds_df.to_csv(os.path.join(predict_results_path, "prediction.csv"), encoding="utf_8_sig", index=False)

        return




def main():
    pass

if __name__ == '__main__':
    main()
