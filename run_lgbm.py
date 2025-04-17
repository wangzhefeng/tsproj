# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pred_power.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-17
# * Version     : 1.0.121713
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
# ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import datetime
import warnings
from typing import Dict

import pandas as pd

from data_provider.data_config import config
from data_provider.data_load import DataLoad
from data_provider.data_preprocess import (
    calc_cabinet_active_power, 
    calc_ups_active_power,
)
from exp.exp_forecasting_lgbm import Model
from utils.plot_results import plot_cv_predictions
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 数据读取类实例
# ------------------------------
data_loader = DataLoad(building_floor="A3F2")

# ------------------------------
# params
# ------------------------------
# input info
pred_method = "multip-step-directly"                                           # 预测方法
freq = "1h"                                                                    # 数据频率
target = "load"                                                                # 预测目标变量名称
lags = 0 if pred_method == "multip-step-directly" else 24                      # 滞后特征构建
n_windows = 1                                                                  # cross validation 窗口数量
history_days = 14                                                              # 历史数据天数
predict_days = 1                                                               # 预测未来1天的功率
data_length = 8 * 24 if n_windows > 1 else history_days * 24                   # 训练数据长度
horizon = predict_days * 24                                                    # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
future_time = now_time + datetime.timedelta(days=predict_days)                 # 时间序列未来结束时刻

# room, cabinet_row, cabinet struct info
room_cabinet_row_cabinet_names = {
    room_name: {
        cabinet_row_name: [
            cabinet
            for cabinet in config["city"]["transformer"]["ups"][room_name][cabinet_row_name].keys()
            if cabinet.startswith("cabinet_")
        ]
        for cabinet_row_name in config["city"]["transformer"]["ups"][room_name].keys() 
        if cabinet_row_name.startswith("cabinet_")
    }
    for room_name in config["city"]["transformer"]["ups"].keys()
    if room_name.startswith("room_")
}


def server_power_forecast():
    """
    上一级模型（服务器）预测结果读取
    """
    logger.info(f"\n{'*' * 140}\nServer model predict result load...\n{'*' * 140}")
    server_output = {
        "history_data": None,
        "predict_data": None,
    }
    
    return server_output


def __cabinet_power_line_forecast(data, 
                                  room,
                                  row,
                                  cabinet: str, 
                                  line: str,
                                  model_cfgs: Dict, 
                                  server_future_data, 
                                  server_history_data, 
                                  cabinet_history_data, 
                                  cabinet_predict_data):
    # history data
    history_data = data[["ds", f"{cabinet}-{line}有功功率"]]
    history_data.columns = ["ds", "load"]
    # future data
    future_data = server_future_data
    # model training, validation, predict
    model_ins = Model(
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    )
    pred_df, eval_scores, _ = model_ins.run()
    eval_scores["room-row-cabinet-line"] = f"{room}-row{row}-{cabinet}-line{line}"
    # 历史数据保存
    cabinet_history_data[f"{cabinet}-line_{line}"] = cabinet_history_data["ds"].map(
        history_data.set_index("ds")["load"]
    )
    # 预测数据保存
    cabinet_predict_data[f"{cabinet}-line_{line}"] = pred_df
    
    return cabinet_history_data, cabinet_predict_data, eval_scores


def cabinet_power_forecast(server_data: Dict = None):
    """
    机柜功率预测

    Args:
        future_data (pd.DataFrame, optional): 服务器功率预测信息. Defaults to None.
    """
    logger.info(f"\n{'*' * 140}\nCabinet model train and predict...\n{'*' * 140}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------
    # 数据收集
    # ------------------------------
    # 历史数据和预测数据
    cabinet_output = {}
    # 所有历史数据整理
    cabinet_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # 预测结果
    cabinet_predict_scores = pd.DataFrame()
    # ------------------------------
    # 模型训练、验证、预测
    # ------------------------------
    for room_name, cabinet_row_content in room_cabinet_row_cabinet_names.items():
        cabinet_output[room_name] = {}
        logger.info(f"{room_name}\n{'=' * 91}")
        for cabinet_row_name, cabinet_names in cabinet_row_content.items():
            cabinet_output[room_name][cabinet_row_name] = {"history_data": None, "predict_data": None}
            cabinet_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
            cabinet_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
            logger.info(f"{room_name}-{cabinet_row_name}\n{'-' * 105}")
            for cabinet_name in cabinet_names:
                logger.info(f"{room_name}-{cabinet_row_name}-{cabinet_name}\n{'^' * 117}")
                # geo params
                room = room_name[-3:]
                row = cabinet_row_name[-1]
                cabinet = cabinet_name[-3:]
                # server data
                server_future_data = server_data["predict_data"]
                server_history_data = server_data["history_data"]
                # cabinet data
                data = data_loader.load_cabinet(device = "机柜", room = room, row = row, cabinet = cabinet)
                data = calc_cabinet_active_power(df = data, cabinet=cabinet, line="A1")
                data = calc_cabinet_active_power(df = data, cabinet=cabinet, line="B1")
                data = calc_cabinet_active_power(df = data, cabinet=cabinet, line="A2")
                data = calc_cabinet_active_power(df = data, cabinet=cabinet, line="B2")
                data[f"{cabinet}-A有功功率"] = data[f"{cabinet}-A1有功功率"] + data[f"{cabinet}-A2有功功率"]
                data[f"{cabinet}-B有功功率"] = data[f"{cabinet}-B1有功功率"] + data[f"{cabinet}-B2有功功率"]
                # 保存历史数据
                cabinet_df_temp = data.rename(columns={
                    old_col: f"{room}-{old_col}" for old_col in data.columns if old_col != "ds"
                })
                cabinet_df = cabinet_df.merge(cabinet_df_temp, on = "ds", how = "left")
                # A 路预测
                logger.info(f"{room_name}-{cabinet_row_name}-{cabinet_name} line A\n{'^' * 124}")
                cabinet_history_data, cabinet_predict_data, eval_scores = __cabinet_power_line_forecast(
                    data, room, row, cabinet, "A", model_cfgs, 
                    server_future_data, server_history_data, 
                    cabinet_history_data, cabinet_predict_data
                )
                cabinet_predict_scores = pd.concat([cabinet_predict_scores, eval_scores], axis = 0)
                # B 路预测
                logger.info(f"{room_name}-{cabinet_row_name}-{cabinet_name} line B\n{'^' * 124}")
                cabinet_history_data, cabinet_predict_data, eval_scores = __cabinet_power_line_forecast(
                    data, room, row, cabinet, "B", model_cfgs, 
                    server_future_data, server_history_data, 
                    cabinet_history_data, cabinet_predict_data
                )
                cabinet_predict_scores = pd.concat([cabinet_predict_scores, eval_scores], axis = 0)
                # ------------------------------
                # 输出结果
                # ------------------------------
                cabinet_output[room_name][cabinet_row_name]["history_data"] = cabinet_history_data
                cabinet_output[room_name][cabinet_row_name]["predict_data"] = cabinet_predict_data

    return cabinet_output, cabinet_predict_scores, cabinet_df


def __cabinet_row_power_line_phase_forecast(data,
                                            room,
                                            row,
                                            line,
                                            phase,
                                            model_cfgs, 
                                            cabinet_history_data,
                                            cabinet_future_data, 
                                            cabinet_row_history_data, 
                                            cabinet_row_predict_data):
    # history data
    cabinet_features = [col for col in cabinet_history_data.columns if col.endswith(f"line_{line}")]
    history_data = cabinet_history_data[["ds", f"load_line_{line}_phase_{phase}"] + cabinet_features]
    history_data = history_data.rename(columns = {f"load_line_{line}_phase_{phase}": "load"})
    # future data
    future_data = cabinet_future_data[
        ["ds"] + 
        [col for col in cabinet_future_data.columns 
         if col.endswith(f"line_{line}")]
    ]
    # model training, validation, predict
    model_ins = Model(
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    )
    pred_df, eval_scores, _ = model_ins.run()
    eval_scores["room-row-line-phase"] = f"{room}-row{row}-line{line}-phase{phase}"
    # 历史数据保存
    cabinet_row_history_data[f"{row}-line_{line}-phase_{phase}"] = cabinet_row_history_data["ds"].map(
        data.set_index("ds")[f"进线_{line}相有功功率"]
    )
    # 预测数据保存
    cabinet_row_predict_data[f"{row}-line_{line}-phase_{phase}"] = pred_df
    
    return cabinet_row_history_data, cabinet_row_predict_data, eval_scores


def cabinet_row_power_forecast(cabinet_data: Dict = None):
    """
    列头柜功率预测
    """
    logger.info(f"{'*' * 120}\ncabinet row model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------
    # 数据收集
    # ------------------------------
    cabinet_row_output = {}
    # 所有历史数据
    cabinet_row_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    cabinet_row_predict_scores = pd.DataFrame()
    # ------------------------------
    # 模型训练、验证、预测
    # ------------------------------
    room_cabinet_row_dict = {
        room_name: list(cabinet_row_names.keys())
        for room_name, cabinet_row_names in room_cabinet_row_cabinet_names.items()
    }
    for room_name, cabinet_row_names in room_cabinet_row_dict.items():
        cabinet_row_output[room_name] = {"history_data": None, "predict_data": None}
        logger.info(f"\n{'=' * 100}\n{room_name}\n{'=' * 100}")
        cabinet_row_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
        cabinet_row_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
        for cabinet_row_name in cabinet_row_names:
            logger.info(f"\n{'-' * 80}\n{room_name}-{cabinet_row_name}\n{'-' * 80}")
            # geo params
            room = room_name[-3:]
            row = cabinet_row_name[-1]
            # cabinet data
            cabinet_future_data = cabinet_data[room_name][cabinet_row_name]["predict_data"]
            cabinet_history_data = cabinet_data[room_name][cabinet_row_name]["history_data"]
            # cabinet row data
            data_line_A = data_loader.load_cabinet_row(device = "列头柜", room = room, row = row, line="A")
            data_line_B = data_loader.load_cabinet_row(device = "列头柜", room = room, row = row, line="B")
            cabinet_history_data["load_line_A_phase_A"] = cabinet_history_data["ds"].map(data_line_A.set_index("ds")["进线_A相有功功率"])
            cabinet_history_data["load_line_A_phase_B"] = cabinet_history_data["ds"].map(data_line_A.set_index("ds")["进线_B相有功功率"])
            cabinet_history_data["load_line_A_phase_C"] = cabinet_history_data["ds"].map(data_line_A.set_index("ds")["进线_C相有功功率"])
            cabinet_history_data["load_line_B_phase_A"] = cabinet_history_data["ds"].map(data_line_B.set_index("ds")["进线_A相有功功率"])
            cabinet_history_data["load_line_B_phase_B"] = cabinet_history_data["ds"].map(data_line_B.set_index("ds")["进线_B相有功功率"])
            cabinet_history_data["load_line_B_phase_C"] = cabinet_history_data["ds"].map(data_line_B.set_index("ds")["进线_C相有功功率"])
            # 保存历史数据
            cabinet_row_df_temp_line_A = data_line_A.rename(columns={
                old_col: f"{room}-{row}-A-{old_col}" for old_col in data_line_A.columns if old_col != "ds"
            })
            cabinet_row_df_temp_line_B = data_line_B.rename(columns={
                old_col: f"{room}-{row}-B-{old_col}" for old_col in data_line_B.columns if old_col != "ds"
            })
            cabinet_row_df = cabinet_row_df.merge(cabinet_row_df_temp_line_A, on = "ds", how = "left")
            cabinet_row_df = cabinet_row_df.merge(cabinet_row_df_temp_line_B, on = "ds", how = "left")
            
            # A/B 路 A/B/C 三相功率预测
            for phase in ["A", "B", "C"]:
                cabinet_row_history_data, cabinet_row_predict_data, eval_scores = __cabinet_row_power_line_phase_forecast(
                    data_line_A, room, row, "A", phase, model_cfgs, 
                    cabinet_history_data, cabinet_future_data, 
                    cabinet_row_history_data, cabinet_row_predict_data
                )
                logger.info(f"\n{eval_scores}")
                cabinet_row_predict_scores = pd.concat([cabinet_row_predict_scores, eval_scores], axis = 0)
                
                cabinet_row_history_data, cabinet_row_predict_data, eval_scores = __cabinet_row_power_line_phase_forecast(
                    data_line_B, room, row, "B", phase, model_cfgs, 
                    cabinet_history_data, cabinet_future_data, 
                    cabinet_row_history_data, cabinet_row_predict_data
                )
                logger.info(f"\n{eval_scores}")
                cabinet_row_predict_scores = pd.concat([cabinet_row_predict_scores, eval_scores], axis = 0)
            # 输出结果
            cabinet_row_output[room_name]["history_data"] = cabinet_row_history_data
            cabinet_row_output[room_name]["predict_data"] = cabinet_row_predict_data

    return cabinet_row_output, cabinet_row_predict_scores, cabinet_row_df


def powerdist_it_room_crac_power_forecast(last_level_data: Dict = None):
    """
    配电室空调、机房精密空调模型训练、推理
    """
    logger.info(f"\n{'*' * 120}\npower distribution and it room crac model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------ 
    # power distribution and it_rooom 数据收集
    # ------------------------------
    power_it_room_output = {"history_data": None, "predict_data": None}
    power_it_room_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    power_it_room_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    power_it_room_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # 预测指标
    power_it_room_crac_predict_scores = pd.DataFrame()
    # ------------------------------
    # power distribution and it room data
    # ------------------------------
    # future data
    power_it_future_data = None
    # history data
    power_it_history_data = None
    # ------------------------------
    # geo params
    # ------------------------------
    device = "配电室机房空调"
    # ------------------------------
    # power distribution and it_rooom power dataload
    # ------------------------------
    data_info = {
        "201": {
            "module": "1",
            "idx": "4",
        },
        "202": {
            "module": "1",
            "idx": "5",
        },
        "203": {
            "module": "2",
            "idx": "5",
        },
        "204": {
            "module": "2",
            "idx": "6",
        },
    }
    for room, room_info in data_info.items():
        logger.info(f"\n{'*' * 100}\npower distribution and it room crac model {room}...\n{'*' * 100}")
        data_line_A = data_loader.load_power_dist_it_room_crac(device, room, module=room_info["module"], line="A", idx=room_info["idx"])
        data_line_B = data_loader.load_power_dist_it_room_crac(device, room, module=room_info["module"], line="B", idx=room_info["idx"])
        # 保存历史数据
        power_it_room_df_temp_line_A = data_line_A.rename(columns={
            old_col: f"{room}-A-{old_col}" for old_col in data_line_A.columns if old_col != "ds"
        })
        power_it_room_df_temp_line_B = data_line_B.rename(columns={
            old_col: f"{room}-B-{old_col}" for old_col in data_line_B.columns if old_col != "ds"
        })
        power_it_room_df = power_it_room_df.merge(power_it_room_df_temp_line_A, on = "ds", how = "left")
        power_it_room_df = power_it_room_df.merge(power_it_room_df_temp_line_B, on = "ds", how = "left")
        # logger.info(data_line_A.columns)
        data_map = {
            f'2AP{room_info["module"]}a{room_info["idx"]}': data_line_A, 
            f'2AP{room_info["module"]}b{room_info["idx"]}': data_line_B,
        }
        for data_name, data in data_map.items():
            logger.info(f"\n{'*' * 80}\npower distribution and it room crac model {room}-{data_name}...\n{'*' * 80}")
            for phase in ["A", "B", "C"]:
                logger.info(f"\n{'*' * 50}\npower distribution and it room crac model {room}-{data_name}-{phase}...\n{'*' * 50}")
                # histoy data
                history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
                history_data["load"] = history_data["ds"].map(data.set_index("ds")[f"{phase}_有功功率_KW"])
                # 缺失值插值填充
                history_data = history_data.interpolate()
                # 缺失值删除
                history_data.dropna(inplace=True, ignore_index=True)
                # logger.info(history_data.columns)
                
                # future data
                future_data = None
                # logger.info(future_data.columns)
                
                # model training, validation, predict
                model_ins = Model(
                    model_cfgs=model_cfgs,
                    history_data=history_data,
                    future_data=future_data,
                )
                pred_df, eval_scores, _ = model_ins.run()
                eval_scores["room-row-line-phase"] = f"PD{room}-line{data_name[-2]}-phase{phase}"
                power_it_room_crac_predict_scores = pd.concat([power_it_room_crac_predict_scores, eval_scores], axis = 0)
                logger.info(f"\n{eval_scores}")
                
                # 历史数据保存
                power_it_room_history_data[f"{room}-{data_name}-phase_{phase}"] = power_it_room_history_data["ds"].map(
                    data.set_index("ds")[f"{phase}_有功功率_KW"]
                )
                # 预测数据保存
                power_it_room_predict_data[f"{room}-{data_name}-phase_{phase}"] = pred_df

    # output result
    power_it_room_output["history_data"] = power_it_room_history_data
    power_it_room_output["predict_data"] = power_it_room_predict_data
    
    return power_it_room_output, power_it_room_crac_predict_scores, power_it_room_df


def battery_room_crac_power_forecast(last_level_data: Dict = None):
    """
    电池室空调模型训练、推理
    """
    logger.info(f"\n{'*' * 120}\nbattery room crac model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    } 
    # ------------------------------ 
    # 数据收集
    # ------------------------------
    battery_room_crac_output = {"history_data": None, "predict_data": None}
    battery_room_crac_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    battery_room_crac_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    battery_room_crac_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # 预测指标
    battery_room_crac_predict_scores = pd.DataFrame()
    # ------------------------------
    # UPS data
    # ------------------------------
    # future data
    battery_future_data = None
    # history data
    battery_history_data = None
    # ------------------------------
    # geo params
    # ------------------------------
    device = "电池室空调"
    # ------------------------------
    # transformer power dataload
    # ------------------------------
    data_room201 = data_loader.load_battery_room_crac(device, room="201", module=1, idx = 1)
    data_room202 = data_loader.load_battery_room_crac(device, room="202", module=1, idx = 2)
    data_room203 = data_loader.load_battery_room_crac(device, room="203", module=2, idx = 1)
    data_room204 = data_loader.load_battery_room_crac(device, room="204", module=2, idx = 2)
    # 保存历史数据
    battery_room_crac_df_temp_room201 = data_room201.rename(columns={
        old_col: f"201-{old_col}" for old_col in data_room201.columns if old_col != "ds"
    })
    battery_room_crac_df_temp_room202 = data_room202.rename(columns={
        old_col: f"202-{old_col}" for old_col in data_room202.columns if old_col != "ds"
    })
    battery_room_crac_df_temp_room203 = data_room203.rename(columns={
        old_col: f"203-{old_col}" for old_col in data_room203.columns if old_col != "ds"
    })
    battery_room_crac_df_temp_room204 = data_room204.rename(columns={
        old_col: f"204-{old_col}" for old_col in data_room204.columns if old_col != "ds"
    })
    battery_room_crac_df = battery_room_crac_df.merge(battery_room_crac_df_temp_room201, on = "ds", how = "left")
    battery_room_crac_df = battery_room_crac_df.merge(battery_room_crac_df_temp_room202, on = "ds", how = "left")
    battery_room_crac_df = battery_room_crac_df.merge(battery_room_crac_df_temp_room203, on = "ds", how = "left")
    battery_room_crac_df = battery_room_crac_df.merge(battery_room_crac_df_temp_room204, on = "ds", how = "left")
    data_info = {
        "201-2AP11": data_room201,
        "202-2AP12": data_room202,
        "203-2AP21": data_room203,
        "204-2AP22": data_room204,
    }
    for data_name, data in data_info.items():
        logger.info(f"\n{'*' * 80}\nbattery room crac model {data_name}...\n{'*' * 80}")
        for phase in ["A", "B", "C"]:
            logger.info(f"\n{'*' * 50}\nbattery room crac model {data_name}-{phase}...\n{'*' * 50}")
            # histoy data
            history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
            history_data["load"] = history_data["ds"].map(data.set_index("ds")[f"{phase}_有功功率_KW"])
            # 缺失值插值填充
            history_data = history_data.interpolate()
            # 缺失值删除
            history_data.dropna(inplace=True, ignore_index=True)
            # logger.info(history_data.columns)
            
            # future data
            future_data = None
            # logger.info(future_data.columns)
            
            # model training, validation, predict
            model_ins = Model(
                model_cfgs=model_cfgs,
                history_data=history_data,
                future_data=future_data,
            )
            pred_df, eval_scores, _ = model_ins.run()
            eval_scores["room-phase"] = f"PD{data_name[0:3]}-phase{phase}"
            battery_room_crac_predict_scores = pd.concat([battery_room_crac_predict_scores, eval_scores], axis = 0)
            logger.info(f"\n{eval_scores}")
            
            # 历史数据保存
            battery_room_crac_history_data[f"{data_name}-phase_{phase}"] = battery_room_crac_history_data["ds"].map(
                data.set_index("ds")[f"{phase}_有功功率_KW"]
            )
            # 预测数据保存
            battery_room_crac_predict_data[f"{data_name}-phase_{phase}"] = pred_df
    # ------------------------------
    # 输出结果
    # ------------------------------
    # output result
    battery_room_crac_output["history_data"] = battery_room_crac_history_data
    battery_room_crac_output["predict_data"] = battery_room_crac_predict_data
    
    return battery_room_crac_output, battery_room_crac_predict_scores, battery_room_crac_df


def __usp_output_power_forecast_module1(line, 
                                        phase,
                                        model_cfgs, 
                                        cabinet_rows_room1, cabinet_rows_room2,
                                        data_ups_output, data_ups_output_name,
                                        cabinet_row_history_data_room1, cabinet_row_history_data_room2, 
                                        cabinet_row_future_data_room1, cabinet_row_future_data_room2,
                                        room_history_data, room_predict_data):
    # history data
    history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    history_data["load"] = history_data["ds"].map(
        data_ups_output.set_index("ds")[f"{phase}相有功功率"]
    )
    cabinet_row_history_data_room1_selected = cabinet_row_history_data_room1[
        ["ds"] + 
        [col for col in cabinet_row_history_data_room1.columns 
         if col.endswith(f"{cabinet_rows_room1[0]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[1]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[2]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[3]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[4]}-line_{line}-phase_{phase}")]
    ]
    cabinet_row_history_data_room2_selected = cabinet_row_history_data_room2[
        ["ds"] + 
        [col for col in cabinet_row_history_data_room2.columns 
         if col.endswith(f"{cabinet_rows_room2[0]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room2[1]}-line_{line}-phase_{phase}")]
    ]
    history_data = history_data.merge(cabinet_row_history_data_room1_selected, on = "ds", how = "left")
    history_data = history_data.merge(cabinet_row_history_data_room2_selected, on = "ds", how = "left")
    history_data = history_data.interpolate()  # 缺失值插值填充
    history_data.dropna(inplace=True, ignore_index=True)  # 缺失值删除
    # future data
    future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    cabinet_row_future_data_room1_selected = cabinet_row_future_data_room1[
        ["ds"] + 
        [col for col in cabinet_row_future_data_room1.columns 
         if col.endswith(f"{cabinet_rows_room1[0]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[1]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[2]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[3]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room1[4]}-line_{line}-phase_{phase}")]
    ]
    cabinet_row_future_data_room2_selected = cabinet_row_future_data_room2[
        ["ds"] + 
        [col for col in cabinet_row_future_data_room2.columns
         if col.endswith(f"{cabinet_rows_room2[0]}-line_{line}-phase_{phase}") or
         col.endswith(f"{cabinet_rows_room2[1]}-line_{line}-phase_{phase}")]
    ]
    future_data = future_data.merge(cabinet_row_future_data_room1_selected, on = "ds", how = "left")
    future_data = future_data.merge(cabinet_row_future_data_room2_selected, on = "ds", how = "left")
    # model training, validation, predict
    model_ins = Model(
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    )
    pred_df, eval_scores, _ = model_ins.run()
    eval_scores["room-line-phase"] = f"{data_ups_output_name}-phase{phase}"
    # 历史数据保存
    room_history_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = room_history_data["ds"].map(
        data_ups_output.set_index("ds")[f"{phase}相有功功率"]
    )
    # 预测数据保存
    room_predict_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = pred_df

    return room_history_data, room_predict_data, eval_scores


def __usp_output_power_forecast_module2(line, 
                                        phase,
                                        model_cfgs, 
                                        data_ups_output, data_ups_output_name,
                                        cabinet_row_history_data, cabinet_row_future_data,
                                        room_history_data, room_predict_data):
    # history data
    history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    history_data["load"] = history_data["ds"].map(
        data_ups_output.set_index("ds")[f"{phase}相有功功率"]
    )
    cabinet_row_history_data_room1_selected = cabinet_row_history_data[
        ["ds"] + 
        [col for col in cabinet_row_history_data.columns 
         if col.endswith(f"line_{line}-phase_{phase}")]
    ]
    history_data = history_data.merge(cabinet_row_history_data_room1_selected, on = "ds", how = "left")
    history_data = history_data.interpolate()  # 缺失值插值填充
    history_data.dropna(inplace=True, ignore_index=True)  # 缺失值删除
    # future data
    future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    cabinet_row_future_data_room1_selected = cabinet_row_future_data[
        ["ds"] + 
        [col for col in cabinet_row_future_data.columns 
         if col.endswith(f"line_{line}-phase_{phase}")]
    ]
    future_data = future_data.merge(cabinet_row_future_data_room1_selected, on = "ds", how = "left")
    # model training, validation, predict
    model_ins = Model(
        model_cfgs=model_cfgs,
        history_data=history_data,
        future_data=future_data,
    )
    pred_df, eval_scores, _ = model_ins.run()
    eval_scores["room-line-phase"] = f"{data_ups_output_name}-phase{phase}"
    # 历史数据保存
    room_history_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = room_history_data["ds"].map(
        data_ups_output.set_index("ds")[f"{phase}相有功功率"]
    )
    # 预测数据保存
    room_predict_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = pred_df

    return room_history_data, room_predict_data, eval_scores


def ups_output_power_forecast(cabinet_row_data: Dict = None):
    logger.info(f"\n{'*' * 120}\nUPS output model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------
    # 数据收集
    # ------------------------------
    room_output = {"history_data": None, "predict_data": None}
    room_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    room_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    ups_output_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # 预测指标
    room_predict_scores = pd.DataFrame()
    # ------------------------------
    # model training and predict
    # ------------------------------
    for module, room_name in enumerate(["room_201_202", "room_204_203"]):
        logger.info(f"\n{'-' * 80}\nUPS output model {room_name}...\n{'-' * 80}")
        # ------------------------------
        # geo params
        # ------------------------------
        device = "UPS输出柜"
        room1 = room_name[5:8]  # 201, 203
        room2 = room_name[-3:]  # 202, 204
        # ------------------------------
        # 下层电路的历史数据、预测数据
        # ------------------------------
        # future data
        cabinet_row_future_data_room1 = cabinet_row_data[f"room_{room1}"]["predict_data"]
        cabinet_row_future_data_room2 = cabinet_row_data[f"room_{room2}"]["predict_data"]
        # history data
        cabinet_row_history_data_room1 = cabinet_row_data[f"room_{room1}"]["history_data"]
        cabinet_row_history_data_room2 = cabinet_row_data[f"room_{room2}"]["history_data"]
        # ------------------------------
        # history data
        # ------------------------------
        # data load
        data_line_A_11 = data_loader.load_ups_output(device, room_group=room_name, room=room1, module=module+1, line="a", row_group=1, idx = 1)
        data_line_A_12 = data_loader.load_ups_output(device, room_group=room_name, room=room1, module=module+1, line="a", row_group=1, idx = 2)
        data_line_B_11 = data_loader.load_ups_output(device, room_group=room_name, room=room2, module=module+1, line="b", row_group=1, idx = 1)
        data_line_B_12 = data_loader.load_ups_output(device, room_group=room_name, room=room2, module=module+1, line="b", row_group=1, idx = 2)
        
        data_line_A_21 = data_loader.load_ups_output(device, room_group=room_name, room=room1, module=module+1, line="a", row_group=4, idx = 1)
        data_line_A_22 = data_loader.load_ups_output(device, room_group=room_name, room=room1, module=module+1, line="a", row_group=4, idx = 2)
        data_line_B_21 = data_loader.load_ups_output(device, room_group=room_name, room=room2, module=module+1, line="b", row_group=4, idx = 1)
        data_line_B_22 = data_loader.load_ups_output(device, room_group=room_name, room=room2, module=module+1, line="b", row_group=4, idx = 2)
        
        # 保存历史数据
        ups_output_df_temp_1 = data_line_A_11.rename(columns={
            old_col: f"{room1}-{module+1}-A11-{old_col}" for old_col in data_line_A_11.columns if old_col != "ds"
        })
        ups_output_df_temp_2 = data_line_A_12.rename(columns={
            old_col: f"{room1}-{module+1}-A12-{old_col}" for old_col in data_line_A_12.columns if old_col != "ds"
        })
        ups_output_df_temp_3 = data_line_B_11.rename(columns={
            old_col: f"{room2}-{module+1}-B11-{old_col}" for old_col in data_line_B_11.columns if old_col != "ds"
        })
        ups_output_df_temp_4 = data_line_B_12.rename(columns={
            old_col: f"{room2}-{module+1}-B12-{old_col}" for old_col in data_line_B_12.columns if old_col != "ds"
        })
        ups_output_df_temp_5 = data_line_A_21.rename(columns={
            old_col: f"{room1}-{module+1}-A41-{old_col}" for old_col in data_line_A_21.columns if old_col != "ds"
        })
        ups_output_df_temp_6 = data_line_A_22.rename(columns={
            old_col: f"{room1}-{module+1}-A42-{old_col}" for old_col in data_line_A_22.columns if old_col != "ds"
        })
        ups_output_df_temp_7 = data_line_B_21.rename(columns={
            old_col: f"{room2}-{module+1}-B41-{old_col}" for old_col in data_line_B_21.columns if old_col != "ds"
        })
        ups_output_df_temp_8 = data_line_B_22.rename(columns={
            old_col: f"{room2}-{module+1}-B42-{old_col}" for old_col in data_line_B_22.columns if old_col != "ds"
        })
        ups_output_df = ups_output_df.merge(ups_output_df_temp_1, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_2, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_3, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_4, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_5, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_6, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_7, on = "ds", how = "left")
        ups_output_df = ups_output_df.merge(ups_output_df_temp_8, on = "ds", how = "left")
        # ------------------------------
        # model
        # ------------------------------
        if room_name == "room_201_202":
            # ------------------------------
            # UPS output: 
            # (201PD)2ANU1a1-1 forecast (data_line_A_11)
            # (201PD)2ANU1a1-2 forecast (data_line_A_12)
            # 
            # (202PD)2ANU1b1-1 forecast (data_line_B_11)
            # (202PD)2ANU1b1-2 forecast (data_line_B_12)
            # ------------------------------
            ups_output_data = {
                "(201PD)2ANU1a1-1_201-a1": data_line_A_11,
                "(201PD)2ANU1a1-2_201-a2": data_line_A_12,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module1(
                        "A", 
                        phase,
                        model_cfgs, 
                        ["F", "G", "H", "I", "J"], ["C", "D"],
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_history_data_room2, 
                        cabinet_row_future_data_room1, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
                    
            
            ups_output_data = {
                "(202PD)2ANU1b1-1_202-b1": data_line_B_11,
                "(202PD)2ANU1b1-2-202-b2": data_line_B_12,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line B-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module1(
                        "B", 
                        phase,
                        model_cfgs, 
                        ["F", "G", "H", "I", "J"], ["C", "D"],
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_history_data_room2, 
                        cabinet_row_future_data_room1, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
            # ------------------------------
            # UPS output: 
            # (201PD)2ANU1a4-1 forecast (data_line_A_21)
            # (201PD)2ANU1a4-2 forecast (data_line_A_22)
            # 
            # (202PD)2ANU1b4-1 forecast (data_line_B_21)
            # (202PD)2ANU1b4-2 forecast (data_line_B_22)
            # ------------------------------
            ups_output_data = {
                "(201PD)2ANU1a4-1_201-a3": data_line_A_21,
                "(201PD)2ANU1a4-2_201-a4": data_line_A_22,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module1(
                        "A", 
                        phase,
                        model_cfgs, 
                        ["A", "B", "C", "D", "E"], ["A", "B"],
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_history_data_room2, 
                        cabinet_row_future_data_room1, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
            
            ups_output_data = {
                "(202PD)2ANU1b4-1_202-b3": data_line_B_21,
                "(202PD)2ANU1b4-2_202-b4": data_line_B_22,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module1(
                        "B", 
                        phase,
                        model_cfgs, 
                        ["A", "B", "C", "D", "E"], ["A", "B"],
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_history_data_room2, 
                        cabinet_row_future_data_room1, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
        else:
            # ------------------------------
            # UPS output: 
            # (204PD)2ANU2a1-1 forecast (data_line_A_11)
            # (204PD)2ANU2a1-2 forecast (data_line_A_12)
            # 
            # (203PD)2ANU2b1-1 forecast (data_line_B_11)
            # (203PD)2ANU2b1-2 forecast (data_line_B_12)
            # ------------------------------
            ups_output_data = {
                "(204PD)2ANU2a1-1_204-a1": data_line_A_11,
                "(204PD)2ANU2a1-2_204-a2": data_line_A_12,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module2(
                        "A", 
                        phase,
                        model_cfgs, 
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_future_data_room1,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
            
            ups_output_data = {
                "(203PD)2ANU2b1-1_203-b1": data_line_B_11,
                "(203PD)2ANU2b1-2_203-b2": data_line_B_12,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module2(
                        "B", 
                        phase,
                        model_cfgs, 
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room2, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
            # ------------------------------
            # UPS output: 
            # (204PD)2ANU2a4-1 forecast (data_line_A_21)
            # (204PD)2ANU2a4-2 forecast (data_line_A_22)
            # 
            # (203PD)2ANU2b4-1 forecast (data_line_B_21)
            # (203PD)2ANU2b4-2 forecast (data_line_B_22)
            # ------------------------------
            ups_output_data = {
                "(204PD)2ANU2a4-1_204-a3": data_line_A_21,
                "(204PD)2ANU2a4-2_204-a4": data_line_A_22,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module2(
                        "A", 
                        phase,
                        model_cfgs, 
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room1, cabinet_row_future_data_room1,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
                
            ups_output_data = {
                "(203PD)2ANU2b4-1_203-b3": data_line_B_21,
                "(203PD)2ANU2b4-2_203-b4": data_line_B_22,
            }
            for data_ups_output_name, data_ups_output in ups_output_data.items():
                logger.info(f"\n{'^' * 50}\nups output [line A-{data_ups_output_name}]\n{'^' * 50}")
                for phase in ["A", "B", "C"]:
                    logger.info(f"\n{'^' * 20}\nups output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 20}")
                    room_history_data, room_predict_data, eval_scores = __usp_output_power_forecast_module2(
                        "B", 
                        phase,
                        model_cfgs, 
                        data_ups_output, data_ups_output_name,
                        cabinet_row_history_data_room2, cabinet_row_future_data_room2,
                        room_history_data, room_predict_data
                    )
                    room_predict_scores = pd.concat([room_predict_scores, eval_scores], axis = 0)
                    logger.info(f"\n{eval_scores}")
    # ------------------------------
    # 输出结果
    # ------------------------------
    room_output["history_data"] = room_history_data
    room_output["predict_data"] = room_predict_data

    return room_output, room_predict_scores, ups_output_df


def crac_ups_output_power_forecast(power_dist_it_room_crac_data: Dict = None):
    logger.info(f"\n{'*' * 120}\nCRAC UPS output model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------
    # 数据收集
    # ------------------------------
    crac_ups_output_output = {"history_data": None, "predict_data": None}
    crac_ups_output_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    crac_ups_output_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    crac_ups_output_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # 预测指标
    crac_ups_output_predict_scores = pd.DataFrame()
    # ------------------------------
    # 下层电路的历史数据、预测数据
    # ------------------------------
    # future data
    power_it_room_crac_future_data = power_dist_it_room_crac_data["predict_data"]
    # history data
    power_it_room_crac_history_data = power_dist_it_room_crac_data["history_data"]
    # ------------------------------
    # model training and predict
    # ------------------------------
    for module, room_name in enumerate(["room_201_202", "room_204_203"]):
        logger.info(f"\n{'-' * 80}\nCRAC UPS output model {room_name}...\n{'-' * 80}")
        # ------------------------------
        # geo params
        # ------------------------------
        room = room_name[-3:]  # 202, 203
        # ------------------------------
        # CRAC UPS power 历史数据
        # ------------------------------
        ups_output_data = data_loader.load_ups_output(
            device = "UPS输出柜", room_group = room_name, room = room, 
            module = module + 1, line = "B", row_group = 7, idx = 1
        )
        # 保存历史数据
        crac_ups_output_df_temp = ups_output_data.rename(columns={
            old_col: f"{room}-{old_col}" for old_col in ups_output_data.columns if old_col != "ds"
        })
        crac_ups_output_df = crac_ups_output_df.merge(crac_ups_output_df_temp, on = "ds", how = "left")
        # logger.info(ups_output_data.columns)
        # ------------------------------
        # model training and predict
        # ------------------------------
        if room == "202":
            data_ups_output_name = "(202PD)2ANU1b7-1_202-b5"
        elif room == "203":
            data_ups_output_name = "(203PD)2ANU2b7-1_203-b5"
        for phase in ["A", "B", "C"]:
            logger.info(f"\n{'^' * 40}\nCRAC UPS output [line A-{data_ups_output_name}-{phase} phase]\n{'^' * 40}")
            # history data
            history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
            history_data["load"] = history_data["ds"].map(ups_output_data.set_index("ds")[f"{phase}相有功功率"])
            power_it_room_crac_history_selected = power_it_room_crac_history_data[
                ["ds"] + 
                [col for col in power_it_room_crac_history_data.columns 
                 if col.startswith(room) and col.endswith(f"phase_{phase}")]
            ]
            history_data = history_data.merge(power_it_room_crac_history_selected, on = "ds", how = "left")
            # 缺失值处理
            history_data = history_data.interpolate()  # 缺失值插值填充
            history_data.dropna(inplace=True, ignore_index=True)  # 缺失值删除
            # logger.info(history_data.columns)
            
            # future data
            future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
            power_it_room_crac_future_selected = power_it_room_crac_future_data[
                ["ds"] + 
                [col for col in power_it_room_crac_future_data.columns
                 if col.startswith(room) and col.endswith(f"phase_{phase}")]
            ]
            future_data = future_data.merge(power_it_room_crac_future_selected, on = "ds", how = "left")
            # logger.info(future_data.columns)
            
            # model training, validation, predict
            model_ins = Model(
                model_cfgs=model_cfgs,
                history_data=history_data,
                future_data=future_data,
            )
            pred_df, eval_scores, _ = model_ins.run()
            eval_scores["room-line-chase"] = f"{room}-lineB-phase{phase}"
            crac_ups_output_predict_scores = pd.concat([crac_ups_output_predict_scores, eval_scores], axis = 0)
            # 历史数据保存
            crac_ups_output_history_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = crac_ups_output_history_data["ds"].map(
                ups_output_data.set_index("ds")[f"{phase}相有功功率"]
            )
            # 预测数据保存
            crac_ups_output_predict_data[f"{data_ups_output_name[-6:]}-phase_{phase}"] = pred_df
    # ------------------------------
    # 输出结果
    # ------------------------------
    crac_ups_output_output["history_data"] = crac_ups_output_history_data
    crac_ups_output_output["predict_data"] = crac_ups_output_predict_data

    return crac_ups_output_output, crac_ups_output_predict_scores, crac_ups_output_df


def ups_power_forecast(ups_output_data: Dict = None, crac_ups_output_data: Dict = None):
    """
    UPS 模型训练、推理

    Args:
        ups_output_output_data (Dict, optional): 
            IT room ups output history and predict data. Defaults to None.
        crac_ups_output_output_data (Dict, optional): 
            CRAC ups output history and predict data. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger.info(f"\n{'*' * 120}\nUPS model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------
    #  数据收集
    # ------------------------------
    # UPS 历史数据、预测数据
    ups_output = {"history_data": None, "predict_data": None}
    ups_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    ups_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    ups_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # TODO UPS 预测结果
    ups_predict_scores = pd.DataFrame()
    # ------------------------------
    # 下层电路的历史数据、预测数据
    # ------------------------------
    # future data
    ups_output_future_data = ups_output_data["predict_data"]
    crac_ups_output_future_data = crac_ups_output_data["predict_data"]
    # history data
    ups_output_history_data = ups_output_data["history_data"]
    crac_ups_output_history_data = crac_ups_output_data["history_data"]
    # ------------------------------
    # TODO geo params
    # ------------------------------
    device = "UPS本体"
    # ------------------------------
    # model training and predict
    # ------------------------------
    for module, room_group in enumerate(["room_201_202", "room_204_203"]):
        logger.info(f"\n{'-' * 80}\nUPS model {room_group}\n{'-' * 80}")
        for line, room in {"A": room_group[5:8], "B": room_group[-3:]}.items():  # {"A": 201, "B": 202}, {"A": 204, "B": 203}
            # ups power data
            data_ups_1 = data_loader.load_ups(device, room_group, room, module+1, line, 1)
            data_ups_2 = data_loader.load_ups(device, room_group, room, module+1, line, 2)
            data_ups_3 = data_loader.load_ups(device, room_group, room, module+1, line, 3)
            data_ups_4 = data_loader.load_ups(device, room_group, room, module+1, line, 4)
            data_ups_5 = data_loader.load_ups(device, room_group, room, module+1, line, 5) if room == "202" or room == "203" else None
            if room == "201" or room == "204":
                data_ups_1 = calc_ups_active_power(data_ups_1)
                data_ups_2 = calc_ups_active_power(data_ups_2)
                data_ups_3 = calc_ups_active_power(data_ups_3)
                data_ups_4 = calc_ups_active_power(data_ups_4)
            # 保存历史数据
            ups_df_temp_1 = data_ups_1.rename(columns={
                old_col: f"{room}-{module+1}-{line}1-{old_col}" for old_col in data_ups_1.columns if old_col != "ds"
            })
            ups_df_temp_2 = data_ups_2.rename(columns={
                old_col: f"{room}-{module+1}-{line}2-{old_col}" for old_col in data_ups_2.columns if old_col != "ds"
            })
            ups_df_temp_3 = data_ups_3.rename(columns={
                old_col: f"{room}-{module+1}-{line}3-{old_col}" for old_col in data_ups_3.columns if old_col != "ds"
            })
            ups_df_temp_4 = data_ups_4.rename(columns={
                old_col: f"{room}-{module+1}-{line}4-{old_col}" for old_col in data_ups_4.columns if old_col != "ds"
            })
            if data_ups_5 is not None:
                ups_df_temp_5 = data_ups_5.rename(columns={
                    old_col: f"{room}-{module+1}-{line}5-{old_col}" for old_col in data_ups_5.columns if old_col != "ds"
                })
            ups_df = ups_df.merge(ups_df_temp_1, on = "ds", how = "left")
            ups_df = ups_df.merge(ups_df_temp_2, on = "ds", how = "left")
            ups_df = ups_df.merge(ups_df_temp_3, on = "ds", how = "left")
            ups_df = ups_df.merge(ups_df_temp_4, on = "ds", how = "left")
            if data_ups_5 is not None:
                ups_df = ups_df.merge(ups_df_temp_5, on = "ds", how = "left")
            ups_data_map = {
                f"2GU{module+1}{line}1({room}PD)UPS本体": data_ups_1,
                f"2GU{module+1}{line}2({room}PD)UPS本体": data_ups_2,
                f"2GU{module+1}{line}3({room}PD)UPS本体": data_ups_3,
                f"2GU{module+1}{line}4({room}PD)UPS本体": data_ups_4,
                f"2GU{module+1}{line}5({room}PD)UPS本体": data_ups_5,
            }
            for ups_data_name, ups_data in ups_data_map.items():
                if ups_data is None: continue
                for phase in ["A", "B", "C"]:
                    # histoy data
                    history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
                    if room == "201" or room == "204":
                        history_data["load"] = history_data["ds"].map(ups_data.set_index("ds")[f"{phase}相输出有功功率"])
                    elif room == "202" or room == "203":
                        if ups_data_name[5] != "5":
                            history_data["load"] = history_data["ds"].map(ups_data.set_index("ds")[f"系统{phase}相输出有功功率"])
                        else:
                            history_data["load"] = history_data["ds"].map(ups_data.set_index("ds")[f"{phase}相有功功率"])
                    ups_output_history = ups_output_history_data[
                        ["ds"] + 
                        [col for col in ups_output_history_data.columns
                         if col.startswith(f"{room}-{line.lower()}{ups_data_name[-13]}-phase_{phase}")]
                    ]
                    crac_ups_output_history = crac_ups_output_history_data[
                        ["ds"] + 
                        [col for col in crac_ups_output_history_data.columns
                         if col.startswith(f"{room}-{line.lower()}{ups_data_name[-13]}-phase_{phase}")]
                    ]
                    history_data = history_data.merge(ups_output_history, on = "ds", how = "left")
                    history_data = history_data.merge(crac_ups_output_history, on = "ds", how = "left")
                    # logger.info(history_data.columns)

                    # future data
                    future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
                    ups_output_future = ups_output_future_data[
                        ["ds"] + 
                        [col for col in ups_output_future_data.columns 
                         if col.startswith(f"{room}-{line.lower()}{ups_data_name[-13]}-phase_{phase}")]
                    ]
                    crac_ups_output_future = crac_ups_output_future_data[
                        ["ds"] + 
                        [col for col in crac_ups_output_future_data.columns
                         if col.startswith(f"{room}-{line.lower()}{ups_data_name[-13]}-phase_{phase}")]
                    ]
                    future_data = future_data.merge(ups_output_future, on = "ds", how = "left")
                    future_data = future_data.merge(crac_ups_output_future, on = "ds", how = "left")
                    # logger.info(future_data.columns)
                    
                    # model training, validation, predict
                    model_ins = Model(
                        model_cfgs=model_cfgs,
                        history_data=history_data,
                        future_data=future_data,
                    )
                    pred_df, eval_scores, _ = model_ins.run()
                    eval_scores["room-line-phase"] = f"{ups_data_name}-phase{phase}"
                    logger.info(eval_scores)
                    ups_predict_scores = pd.concat([ups_predict_scores, eval_scores], axis = 0)
                    
                    # 历史数据保存
                    if room == "201" or room == "204":
                        ups_history_data[f"{room}-{line.lower()}{ups_data_name[5]}-phase_{phase}"] = ups_history_data["ds"].map(
                            ups_data.set_index("ds")[f"{phase}相输出有功功率"]
                        )
                    elif room == "202" or room == "203":
                        if ups_data_name[5] != "5":
                            ups_history_data[f"{room}-{line.lower()}{ups_data_name[5]}-phase_{phase}"] = ups_history_data["ds"].map(
                                ups_data.set_index("ds")[f"系统{phase}相输出有功功率"]
                            )
                        else:
                            ups_history_data[f"{room}-{line.lower()}{ups_data_name[5]}-phase_{phase}"] = ups_history_data["ds"].map(
                                ups_data.set_index("ds")[f"{phase}相有功功率"]
                            )
                    # 预测数据保存
                    ups_predict_data[f"{room}-{line.lower()}{ups_data_name[5]}-phase_{phase}"] = pred_df
    # ------------------------------
    # 输出结果
    # ------------------------------
    ups_output["history_data"] = ups_history_data
    ups_output["predict_data"] = ups_predict_data

    return ups_output, ups_predict_scores, ups_df


def transformer_power_forecast(ups_data: Dict = None, 
                               power_dist_it_room_crac_data: Dict = None, 
                               battery_room_crac_data: Dict = None):
    """
    变压器（低压进线）模型训练、推理

    Args:
        ups_output_data (Dict, optional): 
            ups history and predict data. Defaults to None.
        power_dist_it_room_crac_output_data (Dict, optional): 
            power distribution and IT room CRAC history and predict data. Defaults to None.
        battery_room_crac_output (Dict, optional): 
            battery room CRAC history and predict data. Defaults to None.

    Returns:
        Tuple[Dict, pd.DataFrame]: transformer history and predict data
    """
    logger.info(f"\n{'*' * 120}\nTransformer model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    }
    # ------------------------------ 
    # 数据收集
    # ------------------------------
    # transformer 历史数据、预测数据
    transformer_output = {"history_data": None, "predict_data": None}
    transformer_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    transformer_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # 所有历史数据
    transformer_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # transformer 预测结果
    transformer_predict_scores = pd.DataFrame()
    cv_plots_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # ------------------------------
    # 下层电路的历史数据、预测数据
    # ------------------------------
    # UPS data(UPS本体)
    ups_future_data = ups_data["predict_data"]
    ups_history_data = ups_data["history_data"]
    # power distribution room and IT room CRAC data(配电室空调、机房精密空调)
    power_dist_it_room_crac_future_data = power_dist_it_room_crac_data["predict_data"]
    power_dist_it_room_crac_history_data = power_dist_it_room_crac_data["history_data"]
    # battery room CRAC data(电池室空调)
    battery_room_crac_future_data = battery_room_crac_data["predict_data"]
    battery_room_crac_history_data = battery_room_crac_data["history_data"]
    # ------------------------------
    # TODO geo params
    # ------------------------------
    device = "低压进线"
    pd_rooms = ["201", "202", "203", "204"]
    # ------------------------------
    # transformer power 历史数据
    # ------------------------------
    data_room201_line_A = data_loader.load_transformer(device, room=pd_rooms[0], module=1, line="A")
    data_room202_line_B = data_loader.load_transformer(device, room=pd_rooms[1], module=1, line="B")
    data_room203_line_B = data_loader.load_transformer(device, room=pd_rooms[2], module=2, line="B")
    data_room204_line_A = data_loader.load_transformer(device, room=pd_rooms[3], module=2, line="A")
    # 保存历史数据
    transformer_df_temp_1 = data_room201_line_A.rename(columns={
        old_col: f"{pd_rooms[0]}-{old_col}" for old_col in data_room201_line_A.columns if old_col != "ds"
    })
    transformer_df_temp_2 = data_room202_line_B.rename(columns={
        old_col: f"{pd_rooms[1]}-{old_col}" for old_col in data_room202_line_B.columns if old_col != "ds"
    })
    transformer_df_temp_3 = data_room203_line_B.rename(columns={
        old_col: f"{pd_rooms[2]}-{old_col}" for old_col in data_room203_line_B.columns if old_col != "ds"
    })
    transformer_df_temp_4 = data_room204_line_A.rename(columns={
        old_col: f"{pd_rooms[3]}-{old_col}" for old_col in data_room204_line_A.columns if old_col != "ds"
    })
    transformer_df = transformer_df.merge(transformer_df_temp_1, on = "ds", how = "left")
    transformer_df = transformer_df.merge(transformer_df_temp_2, on = "ds", how = "left")
    transformer_df = transformer_df.merge(transformer_df_temp_3, on = "ds", how = "left")
    transformer_df = transformer_df.merge(transformer_df_temp_4, on = "ds", how = "left")
    # ------------------------------
    # model training and predict
    # ------------------------------
    # TODO 数据遍历形式
    for data_name, data in {
        "201-2AN1a1": data_room201_line_A,
        "202-2AN1b1": data_room202_line_B,
        "203-2AN2b1": data_room203_line_B,
        "204-2AN2a1": data_room204_line_A,
    }.items():
        # TODO
        power_module = data_name.split("-")[1][-3]
        power_dist_room = data_name.split("-")[0]
        power_line = data_name.split("-")[1][-2]
        # 三相电功率预测
        logger.info(f"\n{'^' * 80}\nTransformer model [{data_name}]\n{'^' * 80}")
        for phase in ["A", "B", "C"]:
            logger.info(f"\n{'^' * 40}\nTransformer model [{data_name}-phase{phase}]\n{'^' * 40}")
            # history data
            history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
            history_data["load"] = history_data["ds"].map(data.set_index("ds")[f"{phase}相有功功率"])
            ups_history_selected = ups_history_data[
                ["ds"] + 
                [col for col in ups_history_data.columns 
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}")]
            ]
            battery_room_crac_history_selected = battery_room_crac_history_data[
                ["ds"] + 
                [col for col in battery_room_crac_history_data.columns
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}")]
            ]
            power_dist_it_room_crac_history_selected = power_dist_it_room_crac_history_data[
                ["ds"] + 
                [col for col in power_dist_it_room_crac_history_data.columns 
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}") and data_name[0:3] in ["201", "204"]]
            ]
            history_data = history_data.merge(ups_history_selected, on = "ds", how = "left")
            history_data = history_data.merge(battery_room_crac_history_selected, on = "ds", how = "left")
            history_data = history_data.merge(power_dist_it_room_crac_history_selected, on = "ds", how = "left")
            # logger.info(history_data.columns)
            
            # future data
            future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
            ups_future_selected = ups_future_data[
                ["ds"] + 
                [col for col in ups_history_data.columns
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}")]
            ]
            battery_room_crac_future_selected = battery_room_crac_future_data[
                ["ds"] + 
                [col for col in battery_room_crac_future_data.columns
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}")]
            ]
            power_dist_it_room_crac_future_selected = power_dist_it_room_crac_future_data[
                ["ds"] + 
                [col for col in power_dist_it_room_crac_future_data.columns 
                 if col.startswith(f"{data_name[0:3]}") and col.endswith(f"phase_{phase}") and data_name[0:3] in ["201", "204"]]
            ]
            future_data = future_data.merge(ups_future_selected, on = "ds", how = "left")
            future_data = future_data.merge(battery_room_crac_future_selected, on = "ds", how = "left")
            future_data = future_data.merge(power_dist_it_room_crac_future_selected, on = "ds", how = "left")
            # logger.info(future_data.columns)
            
            # model training, validation, predict
            model_ins = Model(
                model_cfgs=model_cfgs,
                history_data=history_data,
                future_data=future_data,
            )
            pred_df, eval_scores, cv_plot_df = model_ins.run()
            # eval_scores
            eval_scores["room-line-phase"] = f"{data_name}-phase{phase}"
            transformer_predict_scores = pd.concat([transformer_predict_scores, eval_scores], axis = 0)
            # logger.info(f"\n{eval_scores}")
            logger.info(f"cross validation average scores: \n{eval_scores}")

            # pred plot
            cv_plots_df["train_start"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["train_start"])
            cv_plots_df["cutoff"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["cutoff"])
            cv_plots_df["valid_end"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["valid_end"])
            cv_plots_df[f"{data_name}-phase{phase}-Y_pred"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["Y_preds"])
            
            # 数据保存
            metric_name = f"{data_name[0:3]}-line{data_name[-2]}-phase{phase}"
            # transformer history data
            transformer_history_data[metric_name] = transformer_history_data["ds"].map(
                data.set_index("ds")[f"{phase}相有功功率"]
            )
            # transformer predict data
            assert len(pred_df) == len(transformer_predict_data), "The length of pred_df is not equal to transformer_predict_data."
            transformer_predict_data[metric_name] = pred_df
        # TODO 总功率预测
    # ------------------------------
    # 输出结果
    # ------------------------------
    # history, predict data
    transformer_output["history_data"] = transformer_history_data
    transformer_output["predict_data"] = transformer_predict_data
    # 输出结果处理
    cv_plots_df = cv_plots_df.dropna()
    cv_plots_df.set_index("ds", inplace=True)

    return transformer_output, transformer_predict_scores, cv_plots_df, transformer_df


def transformer_power_forecast_total(ups_data: Dict = None, 
                                     power_dist_it_room_crac_data: Dict = None, 
                                     battery_room_crac_data: Dict = None):
    """
    变压器（低压进线）模型训练、推理

    Args:
        ups_output_data (Dict, optional): 
            ups history and predict data. Defaults to None.
        power_dist_it_room_crac_output_data (Dict, optional): 
            power distribution and IT room CRAC history and predict data. Defaults to None.
        battery_room_crac_output (Dict, optional): 
            battery room CRAC history and predict data. Defaults to None.

    Returns:
        Tuple[Dict, pd.DataFrame]: transformer history and predict data
    """
    logger.info(f"\n{'*' * 120}\nTransformer model train and predict...\n{'*' * 120}")
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        "pred_method": pred_method,
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
        },
        "data_length": data_length,
        "horizon": horizon,
        "freq": freq,
        "lags": lags,
        "target": target,
        "n_windows": n_windows,
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 39,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        },
    } 
    # ------------------------------ 
    # 数据收集
    # ------------------------------
    # transformer 历史数据、预测数据
    transformer_output = {"history_data": None, "predict_data": None}
    transformer_history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    transformer_predict_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
    # transformer 预测结果
    transformer_predict_scores = pd.DataFrame()
    cv_plots_df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
    # ------------------------------
    # 下层电路的历史数据、预测数据
    # ------------------------------
    # UPS data(UPS本体)
    ups_future_data = ups_data["predict_data"]
    ups_history_data = ups_data["history_data"]
    # power distribution room and IT room CRAC data(配电室空调、机房精密空调)
    power_dist_it_room_crac_future_data = power_dist_it_room_crac_data["predict_data"]
    power_dist_it_room_crac_history_data = power_dist_it_room_crac_data["history_data"]
    # battery room CRAC data(电池室空调)
    battery_room_crac_future_data = battery_room_crac_data["predict_data"]
    battery_room_crac_history_data = battery_room_crac_data["history_data"]
    # ------------------------------
    # TODO geo params
    # ------------------------------
    device = "低压进线"
    pd_rooms = ["201", "202", "203", "204"]
    # ------------------------------
    # transformer power 历史数据
    # ------------------------------
    data_room201_line_A = data_loader.load_transformer(device, room=pd_rooms[0], module=1, line="A")
    data_room202_line_B = data_loader.load_transformer(device, room=pd_rooms[1], module=1, line="B")
    data_room203_line_B = data_loader.load_transformer(device, room=pd_rooms[2], module=2, line="B")
    data_room204_line_A = data_loader.load_transformer(device, room=pd_rooms[3], module=2, line="A")
    # ------------------------------
    # model training and predict
    # ------------------------------
    # TODO 数据遍历形式
    for data_name, data in {
        "201-2AN1a1": data_room201_line_A,
        "202-2AN1b1": data_room202_line_B,
        "203-2AN2b1": data_room203_line_B,
        "204-2AN2a1": data_room204_line_A
    }.items():
        # TODO
        power_module = data_name.split("-")[1][-3]
        power_dist_room = data_name.split("-")[0]
        power_line = data_name.split("-")[1][-2]
        # 三相电功率预测
        logger.info(f"\n{'^' * 80}\nTransformer model [{data_name}]\n{'^' * 80}")
        data["total_load"] = data["A相有功功率"] + data["B相有功功率"] + data["C相有功功率"]
        # history data
        history_data = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
        history_data["load"] = history_data["ds"].map(data.set_index("ds")["total_load"])
        ups_history_selected = ups_history_data[
            ["ds"] + 
            [col for col in ups_history_data.columns if col.startswith(f"{data_name[0:3]}")]
        ]
        battery_room_crac_history_selected = battery_room_crac_history_data[
            ["ds"] + 
            [col for col in battery_room_crac_history_data.columns if col.startswith(f"{data_name[0:3]}")]
        ]
        power_dist_it_room_crac_history_selected = power_dist_it_room_crac_history_data[
            ["ds"] + 
            [col for col in power_dist_it_room_crac_history_data.columns
             if col.startswith(f"{data_name[0:3]}") and data_name[0:3] in ["201", "204"]]
        ]
        history_data = history_data.merge(ups_history_selected, on = "ds", how = "left")
        history_data = history_data.merge(battery_room_crac_history_selected, on = "ds", how = "left")
        history_data = history_data.merge(power_dist_it_room_crac_history_selected, on = "ds", how = "left")
        # logger.info(history_data.columns)
        
        # future data
        future_data = pd.DataFrame({"ds": pd.date_range(now_time, future_time, freq = freq, inclusive="left")})
        ups_future_selected = ups_future_data[
            ["ds"] + 
            [col for col in ups_history_data.columns if col.startswith(f"{data_name[0:3]}")]
        ]
        battery_room_crac_future_selected = battery_room_crac_future_data[
            ["ds"] + 
            [col for col in battery_room_crac_future_data.columns if col.startswith(f"{data_name[0:3]}")]
        ]
        power_dist_it_room_crac_future_selected = power_dist_it_room_crac_future_data[
            ["ds"] + 
            [col for col in power_dist_it_room_crac_future_data.columns 
             if col.startswith(f"{data_name[0:3]}") and data_name[0:3] in ["201", "204"]]
        ]
        future_data = future_data.merge(ups_future_selected, on = "ds", how = "left")
        future_data = future_data.merge(battery_room_crac_future_selected, on = "ds", how = "left")
        future_data = future_data.merge(power_dist_it_room_crac_future_selected, on = "ds", how = "left")
        # logger.info(future_data.columns)
        
        # model training, validation, predict
        model_ins = Model(
            model_cfgs=model_cfgs,
            history_data=history_data,
            future_data=future_data,
        )
        pred_df, eval_scores, cv_plot_df = model_ins.run()
        # eval scores
        eval_scores["room-line-phase"] = f"{data_name}"
        transformer_predict_scores = pd.concat([transformer_predict_scores, eval_scores], axis = 0)
        logger.info(f"\n{eval_scores}")

        # pred plot
        cv_plots_df["train_start"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["train_start"])
        cv_plots_df["cutoff"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["cutoff"])
        cv_plots_df["valid_end"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["valid_end"])
        cv_plots_df[f"{data_name}-Y_pred"] = cv_plots_df["ds"].map(cv_plot_df.set_index("ds")["Y_preds"])
        
        # 数据保存
        metric_name = f"{data_name[0:3]}-line{data_name[-2]}"
        # transformer history data
        transformer_history_data[metric_name] = transformer_history_data["ds"].map(
            data.set_index("ds")["total_load"]
        )
        # transformer predict data
        transformer_predict_data[metric_name] = pred_df
    # ------------------------------
    # 输出结果
    # ------------------------------
    # history, predict data
    transformer_output["history_data"] = transformer_history_data
    transformer_output["predict_data"] = transformer_predict_data
    # 输出结果处理
    cv_plots_df = cv_plots_df.dropna()
    cv_plots_df.set_index("ds", inplace=True)

    return transformer_output, transformer_predict_scores, cv_plots_df


# TODO
def city_power_forecast(transformer_output_data: Dict = None):
    pass




# 测试代码 main 函数
def main():
    if sys.platform == "win32":
        data_save_path = f"{ROOT}\\dataset\\electricity\\A3F2\\data_A3_201_202_203_204"
    else:
        data_save_path = f"{ROOT}/dataset/electricity/A3F2/data_A3_201_202_203_204"
    logger.info(f"data_save_path: {data_save_path}")

    # server
    server_data = server_power_forecast()
    logger.info(f"\n{server_data}")
    
    # cabinet
    cabinet_data, \
    cabinet_predict_scores, \
    cabinet_df = cabinet_power_forecast(server_data = server_data)
    logger.info(f"\n{cabinet_data}")
    logger.info(f"\n{cabinet_predict_scores}")
    cabinet_df_path = os.path.join(data_save_path, "cabinet_df.csv")
    if not os.path.exists(cabinet_df_path):
        cabinet_df.to_csv(cabinet_df_path)
 
    # cabinet row
    cabinet_row_output_data, \
    cabinet_row_predict_scores, \
    cabinet_row_df = cabinet_row_power_forecast(
        cabinet_data = cabinet_data
    )
    logger.info(f"\n{cabinet_row_output_data}")
    logger.info(f"\n{cabinet_row_predict_scores}")
    cabinet_row_df_path = os.path.join(data_save_path, "cabinet_row_df.csv")
    if not os.path.exists(cabinet_row_df_path):
        cabinet_row_df.to_csv(cabinet_row_df_path)
    
    # power distribution room and it room CRAC
    power_dist_it_room_crac_data, \
    power_it_room_crac_predict_scores, \
    power_it_room_df = powerdist_it_room_crac_power_forecast(
        last_level_data=None
    )
    logger.info(f"\n{power_dist_it_room_crac_data}")
    logger.info(f"\n{power_it_room_crac_predict_scores}")
    power_it_room_df_path = os.path.join(data_save_path, "power_it_room_df.csv")
    if not os.path.exists(power_it_room_df_path):
        power_it_room_df.to_csv(power_it_room_df_path)

    # battery room CRAC
    battery_room_crac_data, \
    battery_room_crac_predict_scores, \
    battery_room_crac_df = battery_room_crac_power_forecast(
        last_level_data=None
    )
    logger.info(f"\n{battery_room_crac_data}")
    logger.info(f"\n{battery_room_crac_predict_scores}")
    battery_room_crac_df_path = os.path.join(data_save_path, "battery_room_crac_df.csv")
    if not os.path.exists(battery_room_crac_df_path):
        battery_room_crac_df.to_csv(battery_room_crac_df_path)
    
    # ups output
    ups_output_data, \
    room_predict_scores, \
    ups_output_df = ups_output_power_forecast(
        cabinet_row_data = cabinet_row_output_data
    )
    logger.info(f"\n{ups_output_data}")
    logger.info(f"\n{room_predict_scores}")
    ups_output_df_path = os.path.join(data_save_path, "ups_output_df.csv")
    if not os.path.exists(ups_output_df_path):
        ups_output_df.to_csv(ups_output_df_path)

    # crac ups output
    crac_ups_output_data, \
    crac_ups_output_predict_scores, \
    crac_ups_output_df = crac_ups_output_power_forecast(
        power_dist_it_room_crac_data = power_dist_it_room_crac_data,
    )
    logger.info(f"\n{crac_ups_output_data}")
    logger.info(f"\n{crac_ups_output_predict_scores}")
    crac_ups_output_df_path = os.path.join(data_save_path, "crac_ups_output_df.csv")
    if not os.path.exists(crac_ups_output_df_path):
        crac_ups_output_df.to_csv(crac_ups_output_df_path)
    
    # ups
    ups_data, ups_predict_scores, ups_df = ups_power_forecast(
        ups_output_data = ups_output_data, 
        crac_ups_output_data = crac_ups_output_data,
    )
    logger.info(f"\n{ups_data}")
    logger.info(f"\n{ups_predict_scores}")
    ups_df_path = os.path.join(data_save_path, "ups_df.csv")
    if not os.path.exists(ups_df_path):
        ups_df.to_csv(ups_df_path)
    
    # TODO transformer
    transformer_data, \
    transformer_predict_scores, \
    transformer_cv_plot_df, \
    transformer_df = transformer_power_forecast(
        ups_data = ups_data,
        power_dist_it_room_crac_data = power_dist_it_room_crac_data,
        battery_room_crac_data = battery_room_crac_data,
    )
    logger.info(f"\n{transformer_data}\n")
    logger.info(f"\n{transformer_predict_scores}\n")
    logger.info(f"\n{transformer_cv_plot_df}")
    plot_cv_predictions(transformer_cv_plot_df, transformer_data, transformer_predict_scores)
    transformer_df_path = os.path.join(data_save_path, "transformer_df.csv")
    if not os.path.exists(transformer_df_path):
        transformer_df.to_csv(transformer_df_path)
    
    # TODO transformer total
    # transformer_data_total, \
    # transformer_predict_scores_total, \
    # transformer_cv_plot_df_total = transformer_power_forecast_total(
    #     ups_data = ups_data,
    #     power_dist_it_room_crac_data = power_dist_it_room_crac_data,
    #     battery_room_crac_data = battery_room_crac_data,
    # )
    # logger.info(f"\n{transformer_data_total}\n")
    # logger.info(f"\n{transformer_predict_scores_total}\n")
    # logger.info(f"\n{transformer_cv_plot_df_total}")
    # plot_cv_predictions(transformer_cv_plot_df_total, transformer_data_total, transformer_predict_scores_total)
    
    # city
    city_elect_data = city_power_forecast(transformer_data = transformer_data)
    logger.info(city_elect_data)

if __name__ == "__main__":
    main()
