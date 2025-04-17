# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_func.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-18
# * Version     : 0.1.071822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "is_weekend",
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.tsa.stattools as ts
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def is_weekend(row: int) -> int:
    """
    判断是否是周末
    
    Args:
        row (int): 一周的第几天

    Returns:
        int: 0: 不是周末, 1: 是周末
    """
    if row == 5 or row == 6:
        return 1
    else:
        return 0


def season(month: int) -> str:
    """
    判断当前月份的季节
    Args:
        day (_type_): _description_

    Returns:
        str: _description_
    """
    pass


def business_season(month: int) -> str:
    """
    业务季度

    Args:
        month (int): _description_

    Returns:
        str: _description_
    """
    pass


def workday_nums(month):
    """
    每个月中的工作日天数

    Args:
        month (_type_): _description_
    """
    pass


def holiday_nums(month):
    """
    每个月中的休假天数

    Args:
        month (_type_): _description_
    """
    pass


def is_summary_time(month):
    """
    是否夏时制

    Args:
        month (_type_): _description_
    """
    pass


def week_of_month(day):
    """
    一月中的第几周

    Args:
        day (_type_): _description_
    """
    pass


def is_holiday(day):
    pass


def holiday_continue_days(holiday):
    """
    节假日连续天数

    Args:
        holiday (_type_): _description_
    """
    pass


def holiday_prev_day_nums(holiday):
    """
    节假日前第 n 天

    Args:
        holiday (_type_): _description_
    """
    pass


def holiday_day_idx(holiday, day):
    """
    节假日第 n 天

    Args:
        holiday (_type_): _description_
        day (_type_): _description_
    """
    pass


def is_tiaoxiu(day):
    """
    是否调休

    Args:
        day (_type_): _description_
    """
    pass


def past_minutes(datetimes):
    """
    一天过去了几分钟

    Args:
        datetimes (_type_): _description_
    """
    pass


def time_period(date):
    """
    一天的哪个时间段

    Args:
        date (_type_): _description_
    """
    pass


def is_work(time):
    """
    该时间点是否营业/上班

    Args:
        time (_type_): _description_
    """
    pass


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Covariance between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_

    Returns:
        float: _description_
    """
    cov_xy = np.cov(x, y)[0][1]

    return cov_xy


def co_integration(x: np.ndarray, y: np.ndarray):
    """
    Co-intergration test between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    r, _, _ = ts.coint(x, y)

    return r


def correlation(x: np.ndarray, y: np.ndarray, method: str = "kendall"):
    """
    Correlation between x and y

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        method (str, optional): _description_. Defaults to "kendall".

    Returns:
        _type_: _description_
    """
    assert method in ["pearson", "spearman", "kendall"]
    corr, p_value = stats.kendalltau(x, y)

    return corr


class FeatureSelect:

    def __init__(self, data, target_name) -> None:
        self.data = data
        self.target_name = target_name

    def features_select_rf(self):
        """
        random forest 特征重要性分析
        """
        # data
        feature_names = self.data.columns.drop([self.target_name])
        
        feature_list = []
        for col in feature_names:
            feature_list.append(np.array(self.data[col]))
        X = np.array(feature_list).T
        y = np.array(np.array(self.data[self.target_name])).reshape(-1, 1)
        # rf
        rf_model = RandomForestRegressor(n_estimators = 500, random_state = 1)
        rf_model.fit(X, y)
        # show importance score
        print(rf_model.feature_importances_)
        # plot importance score
        ticks = [i for i in range(len(feature_names))]
        plt.bar(ticks, rf_model.feature_importances_)
        plt.xticks(ticks, feature_names)
        plt.show()

    def features_select_rfe(self):
        """
        Feature ranking with recursive feature elimination.
        """
        # data
        feature_names = self.data.columns.drop([self.target_name])

        feature_list = []
        for col in feature_names: 
            feature_list.append(np.array(self.data[col]))
        X = np.array(feature_list).T
        y = np.array(np.array(self.data[self.target_name])).reshape(-1, 1)
        # rf
        rfe_model = RFE(
            RandomForestRegressor(n_estimators = 500, random_state = 1), 
            n_features_to_select = 4
        )
        rfe_model.fit(X, y)
        # report selected features
        print('Selected Features:')
        for i in range(len(rfe_model.support_)):
            if rfe_model.support_[i]:
                print(feature_names[i])
        # plot feature rank
        ticks = [i for i in range(len(feature_names))]
        plt.bar(ticks, rfe_model.ranking_)
        plt.xticks(ticks, feature_names)
        plt.show()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
