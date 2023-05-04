# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureSelect.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-23
# * Version     : 0.1.042321
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
