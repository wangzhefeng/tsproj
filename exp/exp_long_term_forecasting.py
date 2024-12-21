import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import metric, DTW
# from utils.dtw_metric import accelerated_dtw, dtw
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        
        return model

    def _get_data(self, flag: str):
        """
        数据集构建

        Args:
            flag (str): 任务类型, ["train", "val", "test"]

        Returns:
            _type_: Dataset, DataLoader
        """
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader 
    
    def _select_criterion(self, loss_name = "MSE"):
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
    
    # TODO
    def _select_criterion_default(self):
        """
        评价指标
        """
        criterion = nn.MSELoss()
        return criterion
    
    def _select_optimizer(self):
        """
        优化器
        """
        model_optim = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.learning_rate
        )
        
        return model_optim 

    def train(self, setting):
        # ------------------------------
        # 数据集构建
        # ------------------------------
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        # ------------------------------
        # checkpoint 保存路径
        # ------------------------------
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # ------------------------------
        # 模型训练
        # ------------------------------
        # time: 模型训练开始时间
        time_now = time.time()
        # 训练数据长度
        train_steps = len(train_loader)
        # 模型优化器
        model_optim = self._select_optimizer()
        # 模型损失函数
        criterion = self._select_criterion(self.args.loss)
        # 早停类实例
        early_stopping = EarlyStopping(patience = self.args.patience, verbose = True)
        # 自动混合精度
        if self.args.use_amp:
            scaler = torch.amp.GradScaler("cuda")
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 每个 epoch 迭代次数记录
            train_loss = []  # 训练误差
            
            # 模型训练
            self.model.train()
            # time: epoch 模型训练开始时间
            epoch_time = time.time()
            # 分 batch 训练
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
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                # encoder-decoder
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                        f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度, -1: 多变量预测单变量，目标特征为最后一个特征， 0: 其他的为所有特征
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 预测 label
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 实际 label
                        # 计算/保存训练损失
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                    f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 预测 label
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 实际 label
                    # 计算/保存损失函数
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                # 日志打印：当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失、
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                # ------------------------------
                # 后向传播、参数优化更新
                # ------------------------------
                if self.args.use_amp:  # 自动混合精度处理
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            # 日志打印：训练 epoch、每个 epoch 训练后的 train_loss、vali_loss、test_loss
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
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

    def vali(self, vali_loader, criterion):
        # 验证损失收集
        total_loss = []
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # ------------------------------
                # 数据预处理
                # ------------------------------ 
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度 
                outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 预测 label
                pred = outputs.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 实际 label
                true = batch_y.detach().cpu()
                # 计算/保存验证损失
                loss = criterion(pred, true)
                total_loss.append(loss) 
        # 计算所有 batch 的平均验证损失
        total_loss = np.average(total_loss)
        # 计算模型输出
        self.model.train()
        
        return total_loss

    def test(self, setting, test = 0):
        # 测试数据集构建
        test_data, test_loader = self._get_data(flag='test')
        # 最优模型加载
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self.best_model_path))
        # 测试数据结果保存地址
        folder_path = os.path.join(self.args.test_results, setting + "/")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # ------------------------------
        # 模型推理
        # ------------------------------ 
        # 模型推理
        self.model.eval()
        # 预测/实际标签
        preds, trues = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # ------------------------------
                # 数据预处理 
                # ------------------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # ------------------------------
                # 前向传播
                # ------------------------------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度 
                outputs = outputs[:, -self.args.pred_len:, :]  # 预测 label
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # 实际 label
                batch_y = batch_y.detach().cpu().numpy()
                # 输出处理
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                pred = outputs[:, :, f_dim:]
                true = batch_y[:, :, f_dim:]
                preds.append(pred)
                trues.append(true)
                # TODO 预测数据可视化
                if i % 20 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        # ------------------------------
        # 预测/实际标签处理
        # ------------------------------
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('after reshape test shape:', preds.shape, trues.shape)
        # ------------------------------
        # 结果保存
        # ------------------------------
        # 计算评价指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dtw = DTW(preds, trues) if self.args.use_dtw else -999
        print(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, dtw:{dtw}")
        # result1 save
        f = open(os.path.join(folder_path, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, dtw:{dtw}")
        f.write('\n')
        f.write('\n')
        f.close()
        # result2 save
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, dtw]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return



# 测试代码 main 函数
def main():
    args = {
        "model": None,         # model name
        "use_gpu": None,       # is use gpu
        "use_multi_gpu": None, # is use multi gpu
        "device_ids": None,    # gpu device ids
        "learning_rate": None,
        "checkpoints": None,
        "patience": None,
        "loss": None,
        "use_amp": None,
        "train_epochs": None,
    }

if __name__ == "__main__":
    main()
