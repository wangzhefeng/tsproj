# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ml_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-04-08
# * Version     : 1.0.040809
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings
import json
import pickle
import traceback
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor

warnings.filterwarnings("ignore")
warnings.simplefilter(action = "ignore", category = FutureWarning)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def preprocess_data_gas_flow(data, col_prefix):
    """
    煤气流量预测模型数据特殊处理
    """
    # 筛选煤气流量大于 10000 的数据
    data = data.loc[data[f"{col_prefix}CQGLMQLL"] > 10000]
    # 计算煤气压力均值
    data[f"{col_prefix}GLMQYL"] = np.nanmean([
        data[f"{col_prefix}GLZMQYL"], 
        data[f"{col_prefix}GLYMQYL"]
    ], axis = 0)
    
    return data


class Config:
    # ------------------------------
    # 本地测试或线上运行
    # ------------------------------
    local_test = True
    # ------------------------------
    # 训练数据、预测数据、模型保存路径
    # ------------------------------
    if sys.platform != "win32":
        # 训练数据保存路径
        training_data_path = "/Users/zfwang/AIMS/projects/boiler/power_plant_gas_boiler/model_training/training_data"
        # 本地测试输入预测数据路径
        predict_data_path = "/Users/zfwang/AIMS/projects/boiler/power_plant_gas_boiler/data/luan_7/data_pred.json"
        # 训练模型保存路径
        model_path = "/Users/zfwang/AIMS/projects/boiler/power_plant_gas_boiler/model_training/trained_models"
    else:
        training_data_path = "D:\\work\\power_plant_gas_boiler\\model_training\\training_data"
        predict_data_path = "D:\\work\\power_plant_gas_boiler\\data\\luan_7\\data_pred.json"
        model_path = "D:\\work\\power_plant_gas_boiler\\model_training\\trained_models"
    # ------------------------------
    # 读取/处理参数
    # ------------------------------
    # 读取算法输入
    params = {}
    # ------------------------------
    # 脚本模型配置
    # ------------------------------
    # 字段名前缀
    col_prefix = "GB_"
    # 过热蒸汽温度
    params[f"{col_prefix}GGCKZQWD"] = np.nanmean([params[f"{col_prefix}GGCKZQWDA"], params[f"{col_prefix}GGCKZQWDB"], params[f"{col_prefix}GGCKZQWDC"]], axis = 0)
    # 过热蒸汽压力
    params[f"{col_prefix}GGCKZQYL"] = np.nanmean([params[f"{col_prefix}GGCKZQYLA"], params[f"{col_prefix}GGCKZQYLB"]], axis = 0)
    # 主给水流量
    params[f"{col_prefix}ZGSLL"] = np.nanmean([params[f"{col_prefix}ZGSLLA"], params[f"{col_prefix}ZGSLLB"]], axis = 0)
    # 空气流量
    params[f"{col_prefix}SFCKLL"] = np.nanmean([params[f"{col_prefix}SFCKLLA"], params[f"{col_prefix}SFCKLLB"]], axis = 0)
    # 汽包压力与主汽压力的压差
    params[f"{col_prefix}QBYL"] = np.nanmean([params[f"{col_prefix}QBYLA"], params[f"{col_prefix}QBYLB"], params[f"{col_prefix}QBYLC"]], axis = 0)
    params[f"{col_prefix}ZQYL"] = np.nanmean([params[f"{col_prefix}ZQYLA"], params[f"{col_prefix}ZQYLB"], params[f"{col_prefix}ZQYLC"]], axis = 0)
    params[f"{col_prefix}QBYL_ZZQYL_diff"] = np.nansum([params[f"{col_prefix}QBYL"], -params[f"{col_prefix}ZQYL"]], axis = 0)
    # 模型配置
    model_cfg = {
        # 煤气流量预测
        "gas_flow": {
            # 训练数据文件路径
            "raw_data_name": "0313-0320-en",
            # 原始数据预处理方法
            "preprocess_func": preprocess_data_gas_flow,
            # 训练数据名称
            "training_data_file": "gas_flow_training.csv",
            # TODO 训练数据中的预测变量
            "predict_vars": [
                f"{col_prefix}CQGLMQLL", f"{col_prefix}GLZMQYL", f"{col_prefix}GLYMQYL",
                f"{col_prefix}SFCKLLA", f"{col_prefix}SFCKLLB",
                f"{col_prefix}ZQLLJSZ",
                f"{col_prefix}QBYLA", f"{col_prefix}QBYLB", f"{col_prefix}QBYLC",
                f"{col_prefix}GGCKZQYLA", f"{col_prefix}GGCKZQYLB",
                f"{col_prefix}ZQYLA", f"{col_prefix}ZQYLB", f"{col_prefix}ZQYLC",
                f"{col_prefix}YLTFYA", f"{col_prefix}YLTFYB", f"{col_prefix}ZLTFYA", f"{col_prefix}ZLTFYB",
                f"{col_prefix}YLTYW", f"{col_prefix}ZLTYW",
                f"{col_prefix}YYHL", f"{col_prefix}ZYHL",
                f"{col_prefix}QQZSMQRKFMFK", f"{col_prefix}QQYSMQRKFMFK", f"{col_prefix}HQZSMQRKFMFK", f"{col_prefix}HQYSMQRKFMFK",
                f"{col_prefix}QQZXMQRKFMFK", f"{col_prefix}QQYXMQRKFMFK", f"{col_prefix}HQZXMQRKFMFK", f"{col_prefix}HQYXMQRKFMFK",
                f"{col_prefix}SFPLFKA", f"{col_prefix}SFPLFKB",
                f"{col_prefix}YFPLFKA", f"{col_prefix}YFPLFKB",
                f"{col_prefix}QBYWA", f"{col_prefix}QBYWB", f"{col_prefix}QBYWC",
                f"{col_prefix}ZGSLLA", f"{col_prefix}ZGSLLB",
                f"{col_prefix}GLGSYL", f"{col_prefix}ZGSFMFK",
                f"{col_prefix}GSBPLFKA", f"{col_prefix}GSBPLFKB",
                f"{col_prefix}GGCKZQWDA", f"{col_prefix}GGCKZQWDB", f"{col_prefix}GGCKZQWDC",
                f"{col_prefix}GRYJJWQWD", f"{col_prefix}GRYJJWHWDA", f"{col_prefix}GRYJJWHWDB",
                f"{col_prefix}GREJJWQWD", f"{col_prefix}GRZEJJWHWDA", f"{col_prefix}GRZEJJWHWDB",
                f"{col_prefix}GRYEJJWHWDA", f"{col_prefix}GRYEJJWHWDB",
                f"{col_prefix}GRYJJWLL", f"{col_prefix}GREJJWLLA", f"{col_prefix}GREJJWLLB",
                f"{col_prefix}GRYJJWFMFK", f"{col_prefix}GREJJWFMFKA", f"{col_prefix}GREJJWFMFKB",
                f"{col_prefix}ZRCKZQWDA", f"{col_prefix}ZRCKZQWDB", f"{col_prefix}ZRCKZQWDC",
                f"{col_prefix}ZRJWLL", f"{col_prefix}ZRJWFMFK",
                f"{col_prefix}YQDBFMFKA", f"{col_prefix}YQDBFMFKB",
                f"{col_prefix}SJGL",
                f"{col_prefix}GGCKZQYL"
            ],
            # 训练数据中的响应变量
            "response_var": f"{col_prefix}CQGLMQLL",
            # 预测模型结果输出名称
            "output_name": "YCMQLL",
            # 模型名称
            "model_name": "LA_01_01_gas_flow",
            # 模型版本
            "model_version": "V1.0",
            # 是否重新训练模型
            "train_model": True,
            # 是否进行模型预测
            "predict_model": True,
            # 时序数据滞后特征
            "lags": 5,
        },
    }
    # ------------------------------
    # 原始数据参数(训练数据)
    # ------------------------------
    # 数据文件名
    data_files = {
        "en": "data/data_en_name.csv",  # 数据根目录/项目数据目录/数据文件名.csv
    }
    # 数据字段名中、英文名称
    data_col_names = {
        "时间": "time",
        "厂区高炉煤气流量": "GB_CQGLMQLL",
    }


def load_data(config: Config,
              data_file_name: str = None, 
              time_col_name = "time",
              upper_name: bool = True,
              columns_map: Dict = {},
              resample: bool = False,
              add_date: bool = False,
              cn_name: bool = False,
              set_index: bool = True,
              prefix: str = "") -> pd.DataFrame:
    """
    数据加载
    """
    # 数据读取
    data = pd.read_csv(
        config.data_files[data_file_name],
        header = 0,
        index_col = False,
        parse_dates = [time_col_name],
        date_parser = lambda dates: pd.to_datetime(dates, format = "%Y-%m-%d %H:%M"),
    )
    # 字段名转换为大写
    if upper_name:
        data.columns = [col.upper() for col in data.columns]
        time_col_name = time_col_name.upper()
    # 字段重命名
    if columns_map:
        for old_col_name, new_col_name in columns_map.items():
            if old_col_name != new_col_name:
                data = data.rename(columns = {old_col_name: new_col_name})
    # 降采样
    if resample:
        data.set_index(time_col_name, drop = True, inplace = True)
        data = data.resample('1min').mean()
        data.reset_index(inplace = True)
    # 增加日期字段
    if add_date:
        data['date'] = data[time_col_name].dt.date.apply(lambda x: str(x))
    # 将字段英文名改为中文名
    if cn_name:
        for zh_name, en_name in config.data_col_names.items():
            data = data.rename(columns = {en_name: zh_name})
    # 设置索引
    if set_index:
        data.set_index(time_col_name, drop = True, inplace = True)
    # 设置字段名前缀
    if prefix:
        data.columns = [f"{prefix}{col}" for col in data.columns]
    
    return data


def build_data(data, col_list: List[str], response_col: str, lags: int):
    """
    时间序列数据整理，构建时序预测模型数据集
    """
    # 筛选变量
    data = data[col_list]
    # 构建时序预测模型数据集
    col_shifted = []
    for col_name in col_list:
        data[f'{col_name}0'] = data[col_name].shift()
        col_shifted.append(f'{col_name}0')
        for i in range(1, lags):
            data[f"{col_name}{i}"] = data[f"{col_name}{i-1}"].shift()
            col_shifted.append(f"{col_name}{i}")
    # 筛选变量
    col_final = [response_col] + col_shifted
    training_data = data[col_final]
    # 重命名
    training_data.columns = ['goal'] + [f"a{i}" for i in range(1, len(col_final))]
    
    return training_data


def model_train_data(config: Config,
                     task: str,
                     raw_data_name: str, 
                     training_data_path: str, 
                     training_data_file: str, 
                     predict_vars: List,
                     response_var: str,
                     lags: int = 5):
    """
    训练数据生成

    Args:
        raw_data_name (str): 原始下载数据名称
        training_data_path (str): 训练数据保存路径
        training_data_file (str): 训练数据保存名称
        predict_vars (List): 预测变量名列表
        response_var (str): 目标变量名
        lags (int): 数据 shift 次数
    """
    # ------------------------------
    # 原始数据读取及处理
    # ------------------------------
    # 数据读取及预处理
    data = load_data(
        config = config,
        data_file_name = raw_data_name,
        time_col_name = "time",
        upper_name = True,
        columns_map = {},
        resample = False,
        add_date = False,
        cn_name = False,
        set_index = False,
        prefix = config.col_prefix,
    )
    
    # 数据特殊处理
    data = config.model_cfg[task]["preprocess_func"](data = data, col_prefix = config.col_prefix)
    # ------------------------------
    # 训练数据构造
    # ------------------------------
    # 构建煤气流量、拱顶温度预测数据集
    training_data = build_data(
        data = data, 
        col_list = predict_vars,
        response_col = response_var,
        lags = lags,
    )
    # ------------------------------
    # 训练数据保存
    # ------------------------------
    output_data_path = f"{training_data_path}/{training_data_file}.csv"
    if not os.path.exists(output_data_path):
        training_data.to_csv(output_data_path, index = False)


def model_train(training_data_path: str,
                training_data_file: str,
                model_path: str, 
                model_name: str, 
                model_version: str):
    """
    模型训练

    Args:
        training_data_path (str): 训练数据文件所在路径
        model_file_path (str): 模型文件保存路径
        model_file_name (str): 模型文件名
        model_file_version (str): 模型文件版本
    """
    # ------------------------------
    # 训练数据读取及处理
    # ------------------------------
    # 训练数据读取
    data = pd.read_csv(f"{training_data_path}/{training_data_file}.csv", index_col = False)
    # 缺失值填充
    data.fillna(0, inplace = True)
    # 训练数据分割
    data_x = data[data.columns[1:]]
    data_y = data[data.columns[0]]
    # ------------------------------
    # 模型训练
    # ------------------------------
    model = ExtraTreeRegressor(random_state = 0)
    model = BaggingRegressor(model, random_state = 0).fit(data_x, data_y)
    # ------------------------------
    # 模型保存
    # ------------------------------
    with open(os.path.join(model_path, f'{model_name}_{model_version}.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    return model


def model_predict(params: Dict, 
                  model_path: str, 
                  model_name: str, 
                  model_version: str, 
                  predict_vars: List,
                  output_name: str,
                  local_test: bool = False,
                  lags: int = 5):
    """
    模型预测
    """
    # ------------------------------
    # 模型加载
    # ------------------------------
    if local_test:
        pre_trained_model = joblib.load(os.path.join(model_path, f"{model_name}_{model_version}.pkl"))
    else:
        pre_trained_model = joblib.load(f"{model_name}_{model_version}.pkl")
    # ------------------------------
    # 模型预测
    # ------------------------------
    # 取数据
    input_data = [params[col] for col in predict_vars]
    # 预测数据构建
    data_list = [col[i] for col in input_data for i in range(lags)]
    columns_num = len(input_data) * lags
    columns_list = [f"a{i}" for i in range(1, columns_num + 1)]
    df = pd.DataFrame(
        np.reshape(a = data_list, newshape = (1, columns_num)),
        columns = columns_list,
    )
    # ------------------------------
    # 模型预测结果
    # ------------------------------
    model_preds = pre_trained_model.predict(df)
    # 预测数据控制
    # ycmq = min(float(LTJMQLL[0]) + MinMQLLRange, ycmq)
    # ycmq = max(float(LTJMQLL[0]) - MinMQLLRange, ycmq)
    # 输出结果
    model_output = {
        output_name: model_preds[0],
    }
    
    return model_output




# 测试代码 main 函数
def main():
    try:
        # ------------------------------
        # 配置类实例
        # ------------------------------
        config = Config()
        # ------------------------------
        # 模型预测
        # ------------------------------
        model_output = {}
        for task in config.model_cfg.keys():
            # 预测任务配置项
            task_cfg = config.model_cfg[task]
            # ------------------------------
            # 训练数据生成、模型训练
            # ------------------------------
            if task_cfg["train_model"]:
                # 训练数据
                model_train_data(
                    config = config,
                    task = task,
                    raw_data_name = task_cfg["raw_data_name"], 
                    training_data_path = config.training_data_path,
                    training_data_file = task_cfg["training_data_file"],
                    predict_vars = task_cfg["predict_vars"],
                    response_var = task_cfg["response_var"],
                    lags = task_cfg["lags"],
                )
                # 训练模型
                model_train(
                    training_data_path = config.training_data_path,
                    training_data_file = task_cfg["training_data_file"],
                    model_path = config.model_path,
                    model_name = task_cfg["model_name"],
                    model_version = task_cfg["model_version"],
                )
            # ------------------------------
            # 模型预测
            # ------------------------------
            if task_cfg["predict_model"]:
                task_output = model_predict(
                    params = config.params,
                    model_path = config.model_path,
                    model_name = task_cfg["model_name"],
                    model_version = task_cfg["model_version"],
                    predict_vars = task_cfg["predict_vars"],
                    output_name = task_cfg["output_name"],
                    local_test = config.local_test,
                    lags = task_cfg["lags"],
                )
                # task_output = {task: pred}
            else:
                pred = config.params[task_cfg["predict_vars"][-1]][0]
                task_output = {task: pred}
            # ------------------------------
            # 结果格式整理
            # ------------------------------
            model_output.update(task_output)
        # ------------------------------
        # 输出结果
        # ------------------------------
        outputs = {}
        outputs['Data'] = model_output
        outputs['LogData'] = [0, 0]
        outputs['Code'] = 1.0
        outputs['Message'] = "success"
        print(json.dumps(outputs))
    except Exception as e:
        # ------------------------------
        # 输出结果
        # ------------------------------
        outputs = {}
        outputs['Data'] = {}
        outputs['LogData'] = [0]
        outputs['Code'] = 0.0
        outputs['Message'] = str(e)
        print(json.dumps(outputs))
        traceback.print_exc()

if __name__ == "__main__":
    main()
