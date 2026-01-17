# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lianghua.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-28
# * Version     : 1.0.092809
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger



# ##############################
# 
# ##############################
class Modelmultilabel():
    def __init__(self, config, input_path=None, ):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)
        
        self.config = config

    def preprocess(self, df_features):
        feature_names = [col for col in df_features.columns if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)]
        # print(feature_names)
        
        final_features = []
        for date, df_tmp in df_features.groupby('date'):
            final_features.append(df_tmp[feature_names].values.ravel())
        
        final_features = np.vstack(final_features)
        return final_features
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)

        x_train = final_features[:-N_valid]
        x_test = final_features[-N_valid:]
        
        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        y_train = df_labels.head(df_labels.shape[0]-N_valid)
        y_test = df_labels.tail(N_valid)

        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=2,
            boosting_type='Plain',
            loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=20,
            random_state=42,
            task_type='GPU',
            reg_lambda=2000
        )
        
        # 训练模型
        model.fit(
            x_train, y_train.fillna(0).values.argsort(-1).argsort(-1)/424,
            eval_set=(x_test, y_test.fillna(0).values.argsort(-1).argsort(-1)/424 ),
            early_stopping_rounds=200
        )

        self.models.append(model)

        predictions = self.predict(df_features)

        pred = y_test.reset_index().copy()
        pred.iloc[:,1:] = predictions[-len(y_test):]
        
        print('score : ', score(y_test.reset_index().tail(N_valid), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)

        predictions = [model.predict(final_features) for model in self.models]
        return np.mean(predictions, 0)

    # def load_model(self, input_path):
    #     self.models.append()

N_valid = 134
config = None
df_features = None
model1 = Modelmultilabel(config, None)
model1.train(df_features, N_valid)

# ##############################
# 
# ##############################
class Modelmultilabel_v2():
    def __init__(self, config, input_path=None, ):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)

        self.config = config

    def preprocess(self, df_features):
        feature_names = [col for col in df_features.columns if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)]
        # print(feature_names)
        
        return df_features[feature_names].values
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()

        x_train = final_features[:-N_valid * id_nunique]
        x_test = final_features[-N_valid * id_nunique:]
        
        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        # y_train = df_labels.head(df_labels.shape[0]-N_valid)
        # y_test = df_labels.tail(N_valid)

        df_features['label1'] = -np.log(df_features.groupby('id')['close'].shift(-1-1)/df_features.groupby('id')['close'].shift(-1))
        df_features['label2'] = -np.log(df_features.groupby('id')['close'].shift(-1-2)/df_features.groupby('id')['close'].shift(-1))
        df_features['label3'] = -np.log(df_features.groupby('id')['close'].shift(-1-3)/df_features.groupby('id')['close'].shift(-1))
        df_features['label4'] = -np.log(df_features.groupby('id')['close'].shift(-1-4)/df_features.groupby('id')['close'].shift(-1))
        y_train = df_features.head(df_features.shape[0]-N_valid * id_nunique)[['label1', 'label2', 'label3', 'label4']].fillna(0)
        y_test = df_features.tail(N_valid * id_nunique)[['label1', 'label2', 'label3', 'label4']].fillna(0)

        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.005,
            depth=3,
            boosting_type='Plain',
            loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=100,
            random_state=42,
            task_type='GPU',
            reg_lambda=2000
        )
        
        # 训练模型
        model.fit(
            x_train, y_train.values,
            eval_set=(x_test, y_test.values ),
            early_stopping_rounds=200
        )
        self.models.append(model)


        predictions = self.predict(df_features)        

        pred = df_labels.tail(N_valid).reset_index().copy()
        
        pred.iloc[:,1:] = predictions[-N_valid:] + np.random.random(predictions[-N_valid:].shape) * 1e-10
        
        print('score : ', score(df_labels.tail(N_valid).reset_index(), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()

        predictions = [model.predict(final_features) for model in self.models]
        predictions = np.mean(predictions, 0).reshape(-1, id_nunique, 4)

        indexlist = [[131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]

        pred_1d = predictions[:, [item[0] for item in indexlist[106*0:106*1]], 0] - predictions[:, [item[1] for item in indexlist[106*0:106*1]], 0] * (np.array([item[1]!=-1 for item in indexlist[106*0:106*1]])).astype('float')
        pred_2d = predictions[:, [item[0] for item in indexlist[106*1:106*2]], 1] - predictions[:, [item[1] for item in indexlist[106*1:106*2]], 1] * (np.array([item[1]!=-1 for item in indexlist[106*1:106*2]])).astype('float')
        pred_3d = predictions[:, [item[0] for item in indexlist[106*2:106*3]], 2] - predictions[:, [item[1] for item in indexlist[106*2:106*3]], 2] * (np.array([item[1]!=-1 for item in indexlist[106*2:106*3]])).astype('float')
        pred_4d = predictions[:, [item[0] for item in indexlist[106*3:106*4]], 3] - predictions[:, [item[1] for item in indexlist[106*3:106*4]], 3] * (np.array([item[1]!=-1 for item in indexlist[106*3:106*4]])).astype('float')

        predictions = np.concatenate([pred_1d, pred_2d, pred_3d, pred_4d,], -1)
        
        return predictions

N_valid = 134
model2 = Modelmultilabel_v2(config, None)
model2.train(df_features, N_valid)

# ##############################
# 
# ##############################
class Modelmultilabel_v3():
    def __init__(self, config, input_path=None, ):
        self.models = []

        if input_path is not None:
            self.load_model(input_path)

        self.config = config

    def preprocess(self, df_features):
        feature_names = [col for col in df_features.columns if (col not in ['date', 'id', 'label_d1', 'high', 'low', 'open', 'interest', 'sprice']) and ('label' not in col)]
        # print(feature_names)
        
        return df_features[feature_names].values
    
    def train(self, df_features, N_valid=134):
        final_features = self.preprocess(df_features)

        id_nunique = df_features.id.nunique()

        df_labels = pd.read_csv(f'{self.config.input_path}/train_labels.csv').set_index('date_id')
        y_train = df_labels.head(df_labels.shape[0]-N_valid).fillna(0).values.argsort(-1).argsort(-1).ravel() / 424
        y_test = df_labels.tail(N_valid).fillna(0).values.argsort(-1).argsort(-1).ravel() / 424


        final_features = final_features.reshape(-1, id_nunique, final_features.shape[-1])

        indexlist = [[131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]

        final_features_list = np.zeros((final_features.shape[0] * len(indexlist), final_features.shape[-1] * 2), dtype='float32')
        c = 0
        from tqdm.auto import tqdm
        for i in tqdm(range(final_features.shape[0])):
            for item in indexlist:
                final_features_list[c, :final_features.shape[-1]] = final_features[i, item[0]] 
                final_features_list[c, final_features.shape[-1]:] = final_features[i, item[1]] * (item[1]!= -1)
            
                c += 1

        x_train = final_features_list[:-N_valid * len(indexlist)]
        x_test = final_features_list[-N_valid * len(indexlist):]

        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=2,
            # boosting_type='Plain',
            # loss_function='MultiRMSE',  # 多输出回归使用MultiRMSE损失
            verbose=100,
            random_state=42,
            task_type='GPU',
            reg_lambda=200
        )
        
        # 训练模型
        model.fit(
            x_train, y_train,
            eval_set=(x_test, y_test ),
            early_stopping_rounds=200
        )
        self.models.append(model)


        predictions = self.predict(df_features)        

        pred = df_labels.tail(N_valid).reset_index().copy()
        
        pred.iloc[:,1:] = predictions[-N_valid:] + np.random.random(predictions[-N_valid:].shape) * 1e-10
        N_test = 500
        
        print('score : ', score(df_labels.tail(N_valid).reset_index(), pred.tail(N_valid), 'date_id'))

    def predict(self, df_features):
        final_features = self.preprocess(df_features)
        id_nunique = df_features.id.nunique()
        final_features = final_features.reshape(-1, id_nunique, final_features.shape[-1])

        indexlist = [[131, -1], [46, 131], [45, 47], [44, 47], [44, 40], [47, 42], [46, 44], [47, 135], [89, 40], [2, 46], [42, 127], [9, 45], [42, 5], [46, 33], [42, 29], [8, 44], [132, 47], [102, 40], [44, 37], [40, 32], [47, 1], [44, 113], [55, 42], [19, 45], [122, 42], [44, 126], [36, 45], [103, 47], [65, 47], [7, 44], [47, 61], [47, 82], [47, 77], [40, 75], [0, 42], [133, 40], [46, 26], [45, 128], [106, 42], [10, 44], [45, 86], [47, 119], [52, 44], [35, 44], [40, 24], [45, 23], [68, 46], [66, 45], [45, 76], [46, 117], [112, 40], [40, 53], [60, 40], [72, 44], [120, 42], [47, 98], [30, 47], [42, 91], [71, 40], [40, 134], [44, 22], [44, 57], [40, 85], [63, 40], [6, 44], [104, 47], [107, 42], [40, 84], [15, 40], [45, 28], [31, 40], [14, 40], [44, 118], [45, 108], [34, 45], [47, 139], [47, 12], [46, 123], [62, 42], [47, 67], [40, 48], [21, 40], [40, 138], [97, 46], [25, 47], [40, 64], [44, 11], [13, 47], [44, 99], [44, 20], [95, 44], [100, 44], [42, 69], [44, 140], [73, 44], [44, 121], [74, 46], [46, 136], [45, 141], [40, 93], [109, 44], [42, 4], [47, 88], [116, 45], [137, 40], [87, 45], [134, -1], [47, 134], [47, 46], [44, 40], [42, 44], [40, 47], [40, 45], [40, 0], [42, 28], [35, 40], [47, 32], [45, 34], [10, 47], [45, 5], [44, 1], [72, 40], [73, 40], [42, 141], [47, 53], [20, 45], [44, 136], [47, 11], [42, 75], [36, 46], [42, 55], [47, 127], [76, 47], [46, 22], [91, 47], [98, 46], [47, 120], [44, 69], [15, 40], [67, 47], [42, 61], [46, 24], [47, 104], [47, 117], [46, 85], [33, 40], [140, 46], [42, 52], [89, 42], [42, 93], [44, 137], [45, 123], [71, 46], [13, 40], [103, 40], [7, 44], [45, 108], [119, 45], [46, 112], [116, 42], [44, 68], [46, 4], [23, 46], [46, 122], [44, 133], [45, 121], [87, 47], [47, 132], [88, 46], [45, 30], [46, 25], [42, 12], [47, 118], [138, 46], [63, 46], [42, 48], [44, 102], [26, 44], [8, 47], [42, 113], [65, 42], [64, 44], [2, 47], [40, 100], [40, 131], [99, 40], [139, 47], [6, 42], [31, 46], [19, 45], [44, 128], [74, 44], [97, 45], [46, 62], [42, 57], [86, 44], [9, 44], [42, 37], [44, 21], [45, 95], [29, 45], [109, 44], [47, 135], [47, 60], [42, 66], [82, 46], [106, 46], [40, 107], [84, 45], [46, 77], [126, 42], [14, 46], [37, -1], [46, 37], [45, 40], [46, 40], [44, 42], [40, 44], [47, 46], [47, 26], [44, 57], [33, 42], [40, 35], [45, 29], [104, 44], [141, 40], [44, 76], [9, 46], [47, 21], [45, 19], [44, 52], [46, 32], [40, 116], [73, 47], [47, 24], [42, 121], [46, 8], [47, 137], [71, 40], [4, 42], [25, 46], [45, 138], [47, 14], [46, 132], [67, 46], [20, 42], [109, 44], [42, 64], [44, 99], [47, 123], [134, 45], [45, 15], [40, 63], [44, 139], [5, 42], [100, 45], [48, 42], [45, 61], [68, 46], [45, 74], [42, 28], [42, 77], [91, 42], [45, 103], [107, 45], [7, 45], [119, 46], [44, 62], [46, 23], [42, 72], [40, 120], [108, 45], [46, 117], [47, 87], [45, 133], [42, 6], [40, 86], [2, 46], [97, 47], [46, 126], [40, 10], [40, 22], [66, 44], [34, 47], [36, 40], [89, 42], [140, 40], [46, 112], [40, 55], [47, 82], [95, 42], [44, 69], [45, 135], [47, 93], [128, 45], [45, 1], [45, 60], [53, 47], [47, 88], [44, 13], [45, 84], [40, 11], [42, 122], [40, 131], [75, 47], [65, 45], [46, 30], [85, 46], [12, 42], [46, 127], [0, 42], [118, 47], [40, 31], [98, 44], [47, 113], [44, 102], [106, 46], [42, 136], [23, -1], [44, 23], [40, 47], [44, 47], [44, 46], [46, 45], [44, 42], [46, 52], [73, 40], [47, 76], [42, 75], [42, 62], [45, 36], [42, 55], [131, 47], [44, 109], [77, 47], [121, 45], [9, 44], [7, 45], [40, 97], [107, 46], [42, 87], [42, 134], [46, 95], [67, 47], [65, 47], [6, 46], [48, 40], [45, 57], [34, 40], [44, 133], [47, 89], [12, 40], [104, 45], [106, 44], [44, 140], [47, 53], [40, 141], [47, 84], [45, 127], [47, 91], [44, 132], [46, 119], [100, 40], [4, 42], [122, 42], [47, 66], [72, 44], [21, 46], [19, 42], [45, 137], [42, 118], [29, 42], [1, 42], [112, 45], [44, 14], [42, 8], [120, 46], [139, 45], [40, 99], [68, 44], [116, 44], [28, 46], [40, 0], [5, 45], [45, 117], [42, 71], [46, 2], [63, 42], [47, 126], [128, 45], [40, 136], [44, 60], [11, 45], [86, 44], [102, 45], [45, 113], [44, 103], [93, 45], [135, 40], [45, 85], [25, 45], [47, 69], [13, 42], [123, 40], [40, 138], [37, 44], [31, 46], [32, 42], [44, 33], [46, 35], [15, 44], [44, 64], [98, 46], [20, 40], [40, 88], [30, 46], [45, 82], [42, 10], [24, 47], [26, 44], [40, 108], [74, 44], [42, 22], [45, 61]]

        final_features_list = np.zeros((final_features.shape[0] * len(indexlist), final_features.shape[-1] * 2), dtype='float32')
        c = 0
        from tqdm.auto import tqdm
        for i in tqdm(range(final_features.shape[0])):
            for item in indexlist:
                final_features_list[c, :final_features.shape[-1]] = final_features[i, item[0]] 
                final_features_list[c, final_features.shape[-1]:] = final_features[i, item[1]] * (item[1]!= -1)
            
                c += 1
        
        id_nunique = df_features.id.nunique()

        predictions = [model.predict(final_features_list) for model in self.models]
        predictions = np.mean(predictions, 0)
        predictions = predictions.reshape(-1, 424)
        
        return predictions


N_valid = 134
model3 = Modelmultilabel_v3(config, None)
model3.train(df_features, N_valid)


# ##############################
# 
# ##############################
prediction1 = model1.predict(df_features).reshape(-1, 424)
prediction2 = model2.predict(df_features).reshape(-1, 424)
prediction3 = model3.predict(df_features).reshape(-1, 424)

pred_ensemble = prediction1+prediction2+prediction3 * 3


df_labels = pd.read_csv(f'{config.input_path}/train_labels.csv').set_index('date_id')
pred = df_labels.tail(N_valid).reset_index().copy()
pred.iloc[:,1:] = pred_ensemble[-N_valid:] + np.random.random(pred_ensemble[-N_valid:].shape) * 1e-10
print('score : ', score(df_labels.tail(N_valid).reset_index().head(90), pred.tail(N_valid).head(90), 'date_id'))
# ##############################
# 
# ##############################
import pandas as pd
import polars as pl
import kaggle_evaluation.mitsui_inference_server
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in ")

global df_train_original
NUM_TARGET_COLUMNS = 424
df_train_original = pd.read_csv(f'{config.input_path}/train.csv')
N_last_day = 60

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    global df_train_original

    test_pandas = test.to_pandas()
    date_id = test_pandas['date_id'].values[0]
    print(date_id)

    df_train_original = df_train_original[df_train_original['date_id'] < date_id]
    df_train_original = pd.concat([df_train_original, test_pandas]).reset_index(drop=True)

    df_processed = preprocess(df_train_original.tail(N_last_day * 143))
    
    df_features = create_features(df_processed).tail(143 * 10)

    # print(df_features.tail())
    
    prediction1 = model1.predict(df_features).reshape(-1, 424)[-1]
    prediction2 = model2.predict(df_features).reshape(-1, 424)[-1]
    prediction3 = model3.predict(df_features).reshape(-1, 424)[-1] * 3
        
    predictions = pl.DataFrame({f'target_{i}': prediction1[i] + prediction2[i] + prediction3[i] for i in range(NUM_TARGET_COLUMNS)})

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == 1
    return predictions


inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
