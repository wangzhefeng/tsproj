# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader_m4.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060916
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from torch.utils.data import Dataset

from utils.ts.m4 import M4Dataset, M4Meta
from utils.log_util import logger


class Dataset_M4(Dataset):
    
    def __init__(self, 
                 args, 
                 root_path,
                 data_path, 
                 flag='pred', 
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='S',
                 target='OT',  
                 freq='15min',
                 timeenc=0, 
                 seasonal_patterns='Yearly',
                 scale=False, 
                 inverse=False,
                 testing_step=None):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        self.flag = flag
        # data size
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse        
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 数据文件
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        
        # split different frequencies
        logger.info(f"debug::dataset.ids: \n{dataset.ids} \ndataset.ids.len: {len(dataset.ids)}")
        logger.info(f"debug::dataset.groups: \n{dataset.groups} \ndataset.groups.len: {len(dataset.groups)}")
        logger.info(f"debug::dataset.frequencies: \n{dataset.frequencies} \ndataset.frequencies.len: {len(dataset.frequencies)}")
        logger.info(f"debug::dataset.horizons: \n{dataset.horizons} \ndataset.horizons.len: {len(dataset.horizons)}")
        logger.info(f"debug::dataset.values[0]: \n{dataset.values[0]} \ndataset.values[0].len: {len(dataset.values[0])}")
        # training_values = np.array([v[~np.isnan(v)] for v in dataset.values[dataset.groups == self.seasonal_patterns]])
        training_values = [v[~np.isnan(v)] for v in dataset.values[dataset.groups == self.seasonal_patterns]]
        self.timeseries = [ts for ts in training_values]
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])

    def __getitem__(self, index):
        # init
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))
        # data split index
        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low = max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high = len(sampled_timeseries),
            size=1
        )[0]
        # data split
        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        
        outsample_window = sampled_timeseries[max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        
        return insample, insample_mask




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
