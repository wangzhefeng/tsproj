# -*- coding: utf-8 -*-

# ***************************************************
# * File        : m4.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-19
# * Version     : 0.1.041902
# * Description : description
# * Link        : M4 Dataset
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = '../dataset/m4') -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz')
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        m4_info = pd.read_csv(info_file)
        return M4Dataset(
            ids = m4_info.M4id.values,
            groups = m4_info.SP.values,
            frequencies = m4_info.Frequency.values,
            horizons = m4_info.Horizon.values,
            values = np.load(train_cache_file if training else test_cache_file, allow_pickle = True)
        )


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


# TODO 未使用
def load_m4_info(INFO_FILE_PATH) -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
