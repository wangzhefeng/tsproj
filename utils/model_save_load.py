# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_deploy.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
# * Description : https://zhuanlan.zhihu.com/p/92691256
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List

import pickle
from sklearn.externals import joblib

from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn_pandas import DataFrameMapper
from pypmml import Model

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ModelDeploy:
    """
    模型部署类
    """
    def __init__(self, save_file_path: str):
        self.save_file_path = save_file_path
    
    def ModelSave(self):
        """
        模型保存
        """
        pass

    def ModelLoad(self):
        """
        模型载入
        """
        pass


class ModelDeployPkl(ModelDeploy):
    """
    模型离线部署类
    """
    def __init__(self, save_file_path: str):
        # 模型保存的目标路径
        self.save_file_path = save_file_path
    
    def save_model(self, model):
        """
        模型保存: 将训练完成的模型保存为pkl文件

        Args:
            model (instance): 模型实例, sklearn机器学习包实例化后训练完毕的模型

        Raises:
            Exception: [description]
        """
        if not self.save_file_path.endswith(".pkl"):
            raise Exception("参数 save_file_path 后缀必须为 'pkl', 请检查.")
        
        with open(self.save_file_path, "wb") as f:
            pickle.dump(model, f, protocol = 2)
        logger.info(f"模型文件已保存至{self.save_file_path}")

    def load_model(self):
        """
        模型加载和使用：载入pkl文件。注意此时预测时列名为['x0', 'x1', ...]

        Raises:
            Exception: [description]

        Returns:
            _type_: sklearn 机器学习包实例类型。预测时用法: model.predict_proba(df[feat_list])[:, 1]
        """
        if not os.path.exists(self.save_file_path):
            raise Exception("参数 save_file_path 指向的文件路径不存在，请检查.")
        
        model = joblib.load(self.save_file_path)
        
        return model


class ModelDeployPmml(ModelDeploy):
    """
    模型在线部署类
    """
    def __init__(self, save_file_path: str):
        self.save_file_path = save_file_path  # 模型保存的目标路径
    
    def save_model(self, model, features_list: List):
        """
        模型保存：将训练完成的模型保存为pmml文件

        Args:
            model (instance): 模型实例，sklearn机器学习包实例化后训练完毕的模型
            features_list (list): 模型特征名称列表（最终入模的特征变量列表），
                若不指定feats_list, 那么写入的pmml中特征名默认取值为['x0', 'x1', ...]

        Raises:
            Exception: [description]
        """
        if not self.save_file_path.endswith(".pmml"):
            raise Exception("参数 save_file_path 后缀必须为 'pmml', 请检查.")

        mapper = DataFrameMapper([
            ([i], None) for i in features_list
        ])
        pipeline = PMMLPipeline([
            ("mapper", mapper),
            ("classifier", model),
        ])
        sklearn2pmml(pipeline, pmml = self.save_file_path)
        logger.info(f"模型文件已保存至{self.save_file_path}")

    def load_model(self):
        """
        模型载入

        Raises:
            Exception: _description_

        Returns:
            _type_: sklearn机器学习包实例类型, 预测时用法: model.predict(json_input_data)
        """
        if not os.path.exists(self.save_file_path):
            raise Exception("参数 save_file_path 指向的文件路径不存在，请检查.")

        model = Model.fromFile(self.save_file_path)

        return model




# 测试代码 main 函数
def main():
    save_file_path = None
    model_deploy_pkl = ModelDeployPkl(save_file_path)
    model_deploy_pmml = ModelDeployPmml(save_file_path)

if __name__ == "__main__":
    main()
