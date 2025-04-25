# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-09
# * Version     : 1.0.120913
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "DataLoad",
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import glob
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DataLoad:
    """
    各层级电力设备数据加载
    """
    
    def __init__(self, building_floor: str):
        if sys.platform == "win32":
            self.data_dir = f"D:\\projects\\tsproj\\dataset\\electricity\\{building_floor}\\"
            # self.data_dir = f"E:\\work\\CES-load_prediction\\dataset\\electricity\\{building_floor}\\"
        else:
            self.data_dir = f"/Users/wangzf/work/CES-load_prediction/dataset/electricity/{building_floor}/"
    
    def load_cabinet(self, device: str = "机柜", room: str = None, row: str = None, cabinet: str = None, line: str = None):
        """
        机房机柜数据加载
        """
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/A3-IT{room}/机柜-IT{room}-RPP-{row}.xlsx"
        )
        # logger.info(data_path)
        # 数据读取
        df = pd.read_excel(data_path)
        # 删除无用的列，并对时间戳列重命名
        df = df.drop(columns = ["序号"]).rename(columns = {"时间": "ds"})
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))
        # 特征筛选
        selected_cols = ["ds"] + [col for col in df.columns if col.startswith(cabinet)]
        df = df[selected_cols]

        return df
    
    def load_cabinet_row(self, device: str = "列头柜", room: str = None, row: str = None, line: str = None):
        """
        机房机柜列数据记载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/A3-IT{room}/列头柜-IT{room}-RPP-{row}{line.lower()}.xlsx"
        )
        df = pd.read_excel(data_path)
        # 删除无用的列，并对时间戳列重命名
        df = df.drop(columns = ["序号"]).rename(columns = {"时间": "ds"})
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))
        # 特征筛选
        if "进线_A相有功功率" not in df.columns:
            df["进线_A相有功功率"] = df["进线_A相电流"] * (df["进线_A相电压"] / 1000) * df["进线_A相功率因数"]
        if "进线_B相有功功率" not in df.columns:
            df["进线_B相有功功率"] = df["进线_B相电流"] * (df["进线_B相电压"] / 1000) * df["进线_B相功率因数"]
        if "进线_C相有功功率" not in df.columns:
            df["进线_C相有功功率"] = df["进线_C相电流"] * (df["进线_C相电压"] / 1000) * df["进线_C相功率因数"]
        # selected_cols = ["ds"] + ["进线_A相有功功率", "进线_B相有功功率", "进线_C相有功功率"]
        # df = df[selected_cols]

        return df

    def load_cabinet_temp_humi(self, device: str = "机房温湿度", room: str = None, row: str = None, line: str = None):
        """
        机房温湿度数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/A3-IT{room}/{room}机房温湿度.xlsx"
        )
        df = pd.read_excel(data_path)
        # 删除无用的列，并对时间戳列重命名
        df = df.drop(columns = ["序号"]).rename(columns = {"时间": "ds"})
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))
        # 特征筛选
        if row == "I":
            selected_cols = ["ds"] + [col for col in df.columns if "HI" in col or "IJ" in col]
        else:
            selected_cols = ["ds"] + [col for col in df.columns if row in col]
        df = df[selected_cols]
        df.columns = [
            "ds", 
            "hc_temp_1", "hc_humi_1", "hc_temp_2", "hc_humi_2",
            "cc_temp_1", "cc_humi_1", "cc_temp_2", "cc_humi_2",
        ]

        return df

    def load_ups_output(self, device: str = "UPS输出柜", room_group: str = None, room: str = None, module: int = None, line: str = None, row_group: int = None, idx: int = None):
        """
        机房数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/{room_group}/({room}PD)2ANU{module}{line.lower()}{row_group}-{idx}.xlsx"
        )
        df = pd.read_excel(data_path, sheet_name="数据表", skiprows=2)
        # 删除无用的行、列，并对时间戳列重命名
        df = df.drop(columns = ["Unnamed: 0"]).rename(columns = {"Unnamed: 1": "ds"})
        df = df.loc[1:, ]
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))

        return df
 
    def load_ups(self, device: str = "UPS本体", room_group: str = None, room: str = None, module: int = None, line: str = None, idx: int = None):
        """
        UPS 本体数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/{room_group}/({room}PD)2GU{module}{line.lower()}{idx}.xlsx"
        )
        df = pd.read_excel(data_path, sheet_name="数据表", skiprows=2)
        # 删除无用的行、列，并对时间戳列重命名
        df = df.drop(columns = ["Unnamed: 0"]).rename(columns = {"Unnamed: 1": "ds"})
        df = df.loc[1:, ]
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))

        return df

    def load_battery_room_crac(self, device: str = "电池室空调", room: str = None, module: int = None, line: str = None, idx: int = None):
        """
        空调数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/{room}电池室2AP{module}{idx}.xlsx"
        )
        df = pd.read_excel(data_path, sheet_name="数据表", skiprows=2)
        # 删除无用的行、列，并对时间戳列重命名
        df = df.drop(columns = ["Unnamed: 0"]).rename(columns = {"Unnamed: 1": "ds"})
        df = df.loc[1:, ]
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))

        return df

    def load_power_dist_it_room_crac(self, device: str = "配电室机房空调", room: str = None, module: int = None, line: str = None, idx: int = None):
        """
        空调数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/room_{room}/{room}配电室2AP{module}{line.lower()}{idx}.xlsx"
        )
        df = pd.read_excel(data_path, sheet_name="数据表", skiprows=2)
        # 删除无用的行、列，并对时间戳列重命名
        df = df.drop(columns = ["Unnamed: 0"]).rename(columns = {"Unnamed: 1": "ds"})
        df = df.loc[1:, ]
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))

        return df
    
    def load_transformer(self, device: str = "低压进线", room: str = None, module: str = "", line: str = "A1"):
        """
        变压器（低压进线）数据加载
        """
        # 数据读取
        data_path = os.path.join(
            self.data_dir, 
            f"{device}/({room}PD)2AN{module}{line.lower()}01.xlsx"
        )
        df = pd.read_excel(data_path, sheet_name="数据表", skiprows=2)
        # 删除无用的行、列，并对时间戳列重命名
        df = df.drop(columns = ["Unnamed: 0"]).rename(columns = {"Unnamed: 1": "ds"})
        df = df.loc[1:, ]
        # 时间戳处理
        df["ds"] = pd.to_datetime(df["ds"].apply(lambda x: x[:-8]))

        return df
    
    # TODO
    def load_diesel_generators(self, device: str = "柴油发电机", line: str = None):
        """
        柴油发电机数据加载
        """
        pass

    # TODO
    def load_city_electricity(self, device: str = "市电", line: str = None):
        """
        市电功率数据加载
        """
        pass

    def load_A3F3_transformer_data(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A3_301_302\\*")
        else:
            data_dir = os.path.join(self.data_dir, "data_A3_301_302/*")
        # params
        freq = "1h"                                                                    # 数据频率
        history_days = 14                                                              # 历史数据天数
        now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
        # 数据收集
        df = pd.DataFrame({"date": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
        # 数据读取与处理
        for idx, file_path in enumerate(glob.glob(data_dir)):
            logger.info(f"Loading file: {file_path}")
            file_df = pd.read_csv(file_path)
            if file_df["value"].dtype == object:
                continue
            else:
                file_df["value"] = file_df["value"].apply(lambda x: np.float32(x))
            file_df["time"] = pd.to_datetime(file_df["time"])
            logger.info(f"Data file: \n{file_df.head()}")
            # 处理目标特征
            file_name = file_path.split("\\")[-1]
            if file_name.startswith("7_0_104_1_612_0"):
                df[f"301_load"] = df["date"].map(file_df.set_index("time")["value"])
            elif file_name.startswith("7_0_300_1_612_0"):
                df[f"302_load"] = df["date"].map(file_df.set_index("time")["value"])
            else:
                df[f"a{idx}"] = df["date"].map(file_df.set_index("time")["value"])
        # 缺失值处理
        df = df.ffill()
        df = df.bfill()
        df = df.dropna(axis=1, how='any', inplace=False)
        
        return df

    def load_A3F2_transformer_data(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A3_201_202_203_204\\*")
        else:
            data_dir = os.path.join(self.data_dir, "data_A3_201_202_203_204/*")
        # params
        freq = "1h"                                                                    # 数据频率
        history_days = 14                                                              # 历史数据天数
        now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
        # 数据收集
        df = pd.DataFrame({"ds": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
        # cabinet df
        df_cabinet = pd.read_csv(os.path.join(data_dir, "cabinet_df.csv"), index_col = 0)
        df_cabinet.columns = ["ds"] + [f"cabinet_{i}" for i in range(1, len(df_cabinet.columns))]
        # cabinet row df
        df_cabinet_row = pd.read_csv(os.path.join(data_dir, "cabinet_row_df.csv"), index_col = 0)
        df_cabinet_row.columns = ["ds"] + [f"cabinet_row_{i}" for i in range(1, len(df_cabinet_row.columns))]
        # power & it room df
        df_power_it_room = pd.read_csv(os.path.join(data_dir, "power_it_room_df.csv"), index_col = 0)
        df_power_it_room.columns = ["ds"] + [f"power_it_room_{i}" for i in range(1, len(df_power_it_room.columns))]
        # battery room crac
        df_battery_room_crac = pd.read_csv(os.path.join(data_dir, "battery_room_crac_df.csv"), index_col = 0)
        df_battery_room_crac.columns = ["ds"] + [f"battery_room_crac_{i}" for i in range(1, len(df_battery_room_crac.columns))]
        # ups output df
        df_ups_output = pd.read_csv(os.path.join(data_dir, "ups_output_df.csv"), index_col = 0)
        df_ups_output.columns = ["ds"] + [f"ups_output_{i}" for i in range(1, len(df_ups_output.columns))]
        # crac ups output df
        df_crac_ups_output = pd.read_csv(os.path.join(data_dir, "crac_ups_output_df.csv"), index_col = 0)
        df_crac_ups_output.columns = ["ds"] + [f"crac_ups_output_{i}" for i in range(1, len(df_crac_ups_output.columns))]
        # ups df
        df_ups = pd.read_csv(os.path.join(data_dir, "ups_df.csv"), index_col = 0)
        df_ups.columns = ["ds"] + [f"ups_{i}" for i in range(1, len(df_ups.columns))]
        # transformer df
        df_transformer = pd.read_csv(os.path.join(data_dir, "transformer_df.csv"), index_col = 0)
        df_transformer["201有功功率"] = df_transformer["201-A相有功功率"] + df_transformer["201-B相有功功率"] + df_transformer["201-C相有功功率"]
        df_transformer["202有功功率"] = df_transformer["202-A相有功功率"] + df_transformer["202-B相有功功率"] + df_transformer["202-C相有功功率"]
        df_transformer["203有功功率"] = df_transformer["203-A相有功功率"] + df_transformer["203-B相有功功率"] + df_transformer["203-C相有功功率"]
        df_transformer["204有功功率"] = df_transformer["204-A相有功功率"] + df_transformer["204-B相有功功率"] + df_transformer["204-C相有功功率"]
        df_transformer = df_transformer.rename(columns = {
            "201有功功率": "201_load",
            "202有功功率": "202_load",
            "203有功功率": "203_load",
            "204有功功率": "204_load",
            "201-A相有功功率": "201_A_load",
            "201-B相有功功率": "201_B_load",
            "201-C相有功功率": "201_C_load",
            "202-A相有功功率": "202_A_load",
            "202-B相有功功率": "202_B_load",
            "202-C相有功功率": "202_C_load",
            "203-A相有功功率": "203_A_load",
            "203-B相有功功率": "203_B_load",
            "203-C相有功功率": "203_C_load",
            "204-A相有功功率": "204_A_load",
            "204-B相有功功率": "204_B_load",
            "204-C相有功功率": "204_C_load",
        }, inplace = False)
        names_map = {
            old_name: f"transformer_{name_idx}" 
            for name_idx, old_name in enumerate(df_transformer.columns) 
            if not old_name.endswith("_load") and old_name != "ds"
        }
        df_transformer = df_transformer.rename(columns = names_map, inplace = False)
        # final df
        df_cabinet["ds"] = pd.to_datetime(df_cabinet["ds"])
        df_cabinet_row["ds"] = pd.to_datetime(df_cabinet_row["ds"])
        df_power_it_room["ds"] = pd.to_datetime(df_power_it_room["ds"])
        df_battery_room_crac["ds"] = pd.to_datetime(df_battery_room_crac["ds"])
        df_ups_output["ds"] = pd.to_datetime(df_ups_output["ds"])
        df_crac_ups_output["ds"] = pd.to_datetime(df_crac_ups_output["ds"])
        df_ups["ds"] = pd.to_datetime(df_ups["ds"])
        df_transformer["ds"] = pd.to_datetime(df_transformer["ds"])
        df = df.merge(df_cabinet, on =  "ds", how = "left")
        df = df.merge(df_cabinet_row, on =  "ds", how = "left")
        df = df.merge(df_power_it_room, on =  "ds", how = "left")
        df = df.merge(df_battery_room_crac, on =  "ds", how = "left")
        df = df.merge(df_ups_output, on =  "ds", how = "left")
        df = df.merge(df_crac_ups_output, on =  "ds", how = "left")
        df = df.merge(df_ups, on =  "ds", how = "left")
        df = df.merge(df_transformer, on =  "ds", how = "left") 
        # 缺失值处理
        df = df.ffill()
        df = df.bfill()
        
        return df

    def load_A1F2_transformer_data(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A1_201_202_203_204\\*")
        else:
            data_dir = os.path.join(self.data_dir, "data_A1_201_202_203_204/*")
        # params
        freq = "1h"                                                                    # 数据频率
        history_days = 14                                                              # 历史数据天数
        now = datetime.datetime(2024, 12, 1, 0, 0, 0)                                  # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)         # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)  # 时间序列历史数据开始时刻
        # 数据收集
        df = pd.DataFrame({"date": pd.date_range(start_time, now_time, freq = freq, inclusive="left")})
        # 数据读取与处理
        for idx, file_path in enumerate(glob.glob(data_dir)):
            logger.info(f"Loading file: {file_path}")
            file_df = pd.read_csv(file_path)
            if file_df["value"].dtype == object:
                continue
            else:
                file_df["value"] = file_df["value"].apply(lambda x: np.float32(x))
            file_df["time"] = pd.to_datetime(file_df["time"])
            logger.info(f"Data file: \n{file_df.head()}")
            # 处理目标特征
            file_name = file_path.split("\\")[-1]
            logger.info(f"file_name: {file_name}")
            if file_name.startswith("46_0_202_1_612_0"):
                df[f"201_load"] = df["date"].map(file_df.set_index("time")["value"])
            elif file_name.startswith("55_0_219_1_612_0"):
                df[f"202_load"] = df["date"].map(file_df.set_index("time")["value"])
            elif file_name.startswith("13_0_202_1_612_0"):
                df[f"203_load"] = df["date"].map(file_df.set_index("time")["value"])
            elif file_name.startswith("3_0_218_1_612_0"):
                df[f"204_load"] = df["date"].map(file_df.set_index("time")["value"])
            else:
                df[f"a{idx}"] = df["date"].map(file_df.set_index("time")["value"])
        # 缺失值处理
        df = df.ffill()
        df = df.bfill()
        df = df.dropna(axis=1, how='any', inplace=False)

        return df

    def get_A1F2_files(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A1_201_202_203_204\\*")
        else:
            data_dir = os.path.join(self.data_dir, "data_A1_201_202_203_204/*")
        # 数据读取与处理 
        df = pd.DataFrame({
            "resource_id": [file_path.split("\\")[-1][:-26] for file_path in glob.glob(data_dir)],
        })
        df.to_csv(os.path.join(self.data_dir, "A1F2_files.csv"), index=False)

    def get_A3F3_files(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A3_301_302\\*")
        else:
            data_dir = os.path.join(self.data_dir, "data_A3_301_302/*")
        # 数据读取与处理 
        df = pd.DataFrame({
            "resource_id": [file_path.split("\\")[-1][:-26] for file_path in glob.glob(data_dir)],
        })
        df.to_csv(os.path.join(self.data_dir, "A3F3_files.csv"), index=False)

    def move_unused_files(self):
        # 所有数据文件所在路径
        if sys.platform == "win32":
            data_dir = os.path.join(self.data_dir, "data_A3_301_302\\*")
            new_data_dir = os.path.join(self.data_dir, "data_A3_303_304\\")

        else:
            data_dir = os.path.join(self.data_dir, "data_A3_301_302/*")
        files_303_304 = [
            "0_878_32",
            "0_878_33",
            "0_878_34",
            "1_0_103_1_682_0",
            "1_0_104_1_652_0",
            "1_0_104_1_653_0",
            "1_0_104_1_654_0",
            "1_0_104_1_655_0",
            "1_0_104_1_656_0",
            "1_0_104_1_657_0",
            "1_0_104_1_658_0",
            "1_0_104_1_659_0",
            "1_0_104_1_682_0",
            "1_0_104_1_776_0",
            "1_0_104_1_788_0",
            "1_0_104_1_800_0",
            "1_0_104_1_812_0",
            "1_0_104_1_824_0",
            "1_0_104_1_836_0",
            "1_0_104_1_848_0",
            "1_0_104_1_860_0",
            "1_0_104_1_872_0",
            "1_0_104_1_884_0",
            "1_0_104_1_896_0",
            "1_0_104_1_908_0",
            "1_0_104_1_920_0",
            "1_0_104_1_932_0",
            "1_0_105_1_652_0",
            "1_0_105_1_653_0",
            "1_0_105_1_654_0",
            "1_0_105_1_655_0",
            "1_0_105_1_656_0",
            "1_0_105_1_657_0",
            "1_0_105_1_658_0",
            "1_0_105_1_659_0",
            "1_0_105_1_682_0",
            "1_0_105_1_776_0",
            "1_0_105_1_788_0",
            "1_0_105_1_800_0",
            "1_0_105_1_812_0",
            "1_0_105_1_824_0",
            "1_0_105_1_836_0",
            "1_0_105_1_848_0",
            "1_0_105_1_860_0",
            "1_0_105_1_872_0",
            "1_0_105_1_884_0",
            "1_0_105_1_896_0",
            "1_0_105_1_908_0",
            "1_0_105_1_920_0",
            "1_0_105_1_932_0",
            "1_0_106_1_652_0",
            "1_0_106_1_653_0",
            "1_0_106_1_654_0",
            "1_0_106_1_655_0",
            "1_0_106_1_656_0",
            "1_0_106_1_657_0",
            "1_0_106_1_658_0",
            "1_0_106_1_659_0",
            "1_0_106_1_682_0",
            "1_0_106_1_776_0",
            "1_0_106_1_788_0",
            "1_0_106_1_800_0",
            "1_0_106_1_812_0",
            "1_0_106_1_824_0",
            "1_0_106_1_836_0",
            "1_0_106_1_848_0",
            "1_0_106_1_860_0",
            "1_0_106_1_872_0",
            "1_0_106_1_884_0",
            "1_0_106_1_896_0",
            "1_0_106_1_908_0",
            "1_0_106_1_920_0",
            "1_0_106_1_932_0",
            "1_0_106_1_944_0",
            "1_0_107_1_652_0",
            "1_0_107_1_653_0",
            "1_0_107_1_654_0",
            "1_0_107_1_655_0",
            "1_0_107_1_656_0",
            "1_0_107_1_657_0",
            "1_0_107_1_658_0",
            "1_0_107_1_659_0",
            "1_0_107_1_682_0",
            "1_0_107_1_776_0",
            "1_0_107_1_788_0",
            "1_0_107_1_800_0",
            "1_0_107_1_812_0",
            "1_0_107_1_824_0",
            "1_0_107_1_836_0",
            "1_0_107_1_848_0",
            "1_0_107_1_860_0",
            "1_0_107_1_872_0",
            "1_0_107_1_884_0",
            "1_0_107_1_896_0",
            "1_0_107_1_908_0",
            "1_0_107_1_920_0",
            "1_0_107_1_932_0",
            "1_0_107_1_944_0",
            "1_0_108_1_652_0",
            "1_0_108_1_653_0",
            "1_0_108_1_654_0",
            "1_0_108_1_655_0",
            "1_0_108_1_656_0",
            "1_0_108_1_657_0",
            "1_0_108_1_658_0",
            "1_0_108_1_659_0",
            "1_0_108_1_682_0",
            "1_0_108_1_776_0",
            "1_0_108_1_788_0",
            "1_0_108_1_800_0",
            "1_0_108_1_812_0",
            "1_0_108_1_824_0",
            "1_0_108_1_836_0",
            "1_0_108_1_848_0",
            "1_0_108_1_860_0",
            "1_0_108_1_872_0",
            "1_0_108_1_884_0",
            "1_0_108_1_896_0",
            "1_0_108_1_908_0",
            "1_0_108_1_920_0",
            "1_0_108_1_932_0",
            "1_0_108_1_944_0",
            "2_0_101_1_652_0",
            "2_0_101_1_653_0",
            "2_0_101_1_654_0",
            "2_0_101_1_655_0",
            "2_0_101_1_656_0",
            "2_0_101_1_657_0",
            "2_0_101_1_658_0",
            "2_0_101_1_659_0",
            "2_0_101_1_682_0",
            "2_0_101_1_776_0",
            "2_0_101_1_788_0",
            "2_0_101_1_800_0",
            "2_0_101_1_812_0",
            "2_0_101_1_824_0",
            "2_0_101_1_836_0",
            "2_0_101_1_848_0",
            "2_0_101_1_860_0",
            "2_0_101_1_872_0",
            "2_0_101_1_884_0",
            "2_0_101_1_896_0",
            "2_0_101_1_908_0",
            "2_0_101_1_920_0",
            "2_0_101_1_932_0",
            "2_0_102_1_652_0",
            "2_0_102_1_653_0",
            "2_0_102_1_654_0",
            "2_0_102_1_655_0",
            "2_0_102_1_656_0",
            "2_0_102_1_657_0",
            "2_0_102_1_658_0",
            "2_0_102_1_659_0",
            "2_0_102_1_682_0",
            "2_0_102_1_776_0",
            "2_0_102_1_788_0",
            "2_0_102_1_800_0",
            "2_0_102_1_812_0",
            "2_0_102_1_824_0",
            "2_0_102_1_836_0",
            "2_0_102_1_848_0",
            "2_0_102_1_860_0",
            "2_0_102_1_872_0",
            "2_0_102_1_884_0",
            "2_0_102_1_896_0",
            "2_0_102_1_908_0",
            "2_0_102_1_920_0",
            "2_0_102_1_932_0",
            "2_0_103_1_652_0",
            "2_0_103_1_653_0",
            "2_0_103_1_654_0",
            "2_0_103_1_655_0",
            "2_0_103_1_656_0",
            "2_0_103_1_657_0",
            "2_0_103_1_658_0",
            "2_0_103_1_659_0",
            "2_0_103_1_682_0",
            "2_0_103_1_776_0",
            "2_0_103_1_788_0",
            "2_0_103_1_800_0",
            "2_0_103_1_812_0",
            "2_0_103_1_824_0",
            "2_0_103_1_836_0",
            "2_0_103_1_848_0",
            "2_0_103_1_860_0",
            "2_0_103_1_872_0",
            "2_0_103_1_884_0",
            "2_0_103_1_896_0",
            "2_0_103_1_908_0",
            "2_0_103_1_920_0",
            "2_0_103_1_932_0",
            "2_0_103_1_944_0",
            "2_0_104_1_652_0",
            "2_0_104_1_653_0",
            "2_0_104_1_654_0",
            "2_0_104_1_655_0",
            "2_0_104_1_656_0",
            "2_0_104_1_657_0",
            "2_0_104_1_658_0",
            "2_0_104_1_659_0",
            "2_0_104_1_682_0",
            "2_0_104_1_776_0",
            "2_0_104_1_788_0",
            "2_0_104_1_800_0",
            "2_0_104_1_812_0",
            "2_0_104_1_824_0",
            "2_0_104_1_836_0",
            "2_0_104_1_848_0",
            "2_0_104_1_860_0",
            "2_0_104_1_872_0",
            "2_0_104_1_884_0",
            "2_0_104_1_896_0",
            "2_0_104_1_908_0",
            "2_0_104_1_920_0",
            "2_0_104_1_932_0",
            "2_0_104_1_944_0",
            "2_0_105_1_652_0",
            "2_0_105_1_653_0",
            "2_0_105_1_654_0",
            "2_0_105_1_655_0",
            "2_0_105_1_656_0",
            "2_0_105_1_657_0",
            "2_0_105_1_658_0",
            "2_0_105_1_659_0",
            "2_0_105_1_682_0",
            "2_0_105_1_776_0",
            "2_0_105_1_788_0",
            "2_0_105_1_800_0",
            "2_0_105_1_812_0",
            "2_0_105_1_824_0",
            "2_0_105_1_836_0",
            "2_0_105_1_848_0",
            "2_0_105_1_860_0",
            "2_0_105_1_872_0",
            "2_0_105_1_884_0",
            "2_0_105_1_896_0",
            "2_0_105_1_908_0",
            "2_0_105_1_920_0",
            "2_0_105_1_932_0",
            "2_0_105_1_944_0",
            "2_0_106_1_652_0",
            "2_0_106_1_653_0",
            "2_0_106_1_654_0",
            "2_0_106_1_655_0",
            "2_0_106_1_656_0",
            "2_0_106_1_657_0",
            "2_0_106_1_658_0",
            "2_0_106_1_659_0",
            "2_0_106_1_682_0",
            "2_0_106_1_776_0",
            "2_0_106_1_788_0",
            "2_0_106_1_800_0",
            "2_0_106_1_812_0",
            "2_0_106_1_824_0",
            "2_0_106_1_836_0",
            "2_0_106_1_848_0",
            "2_0_106_1_860_0",
            "2_0_106_1_872_0",
            "2_0_106_1_884_0",
            "2_0_106_1_896_0",
            "2_0_106_1_908_0",
            "2_0_106_1_920_0",
            "2_0_106_1_932_0",
            "2_0_106_1_944_0",
            "2_0_107_1_652_0",
            "2_0_107_1_653_0",
            "2_0_107_1_654_0",
            "2_0_107_1_655_0",
            "2_0_107_1_656_0",
            "2_0_107_1_657_0",
            "2_0_107_1_658_0",
            "2_0_107_1_659_0",
            "2_0_107_1_682_0",
            "2_0_107_1_776_0",
            "2_0_107_1_788_0",
            "2_0_107_1_800_0",
            "2_0_107_1_812_0",
            "2_0_107_1_824_0",
            "2_0_107_1_836_0",
            "2_0_107_1_848_0",
            "2_0_107_1_860_0",
            "2_0_107_1_872_0",
            "2_0_107_1_884_0",
            "2_0_107_1_896_0",
            "2_0_107_1_908_0",
            "2_0_107_1_920_0",
            "2_0_107_1_932_0",
            "2_0_108_1_652_0",
            "2_0_108_1_653_0",
            "2_0_108_1_654_0",
            "2_0_108_1_655_0",
            "2_0_108_1_656_0",
            "2_0_108_1_657_0",
            "2_0_108_1_658_0",
            "2_0_108_1_659_0",
            "2_0_108_1_682_0",
            "2_0_108_1_776_0",
            "2_0_108_1_788_0",
            "2_0_108_1_800_0",
            "2_0_108_1_812_0",
            "2_0_108_1_824_0",
            "2_0_108_1_836_0",
            "2_0_108_1_848_0",
            "2_0_108_1_860_0",
            "2_0_108_1_872_0",
            "2_0_108_1_884_0",
            "2_0_108_1_896_0",
            "2_0_108_1_908_0",
            "2_0_108_1_920_0",
            "2_0_108_1_932_0",
            "3_0_102_1_652_0",
            "3_0_102_1_653_0",
            "3_0_102_1_654_0",
            "3_0_102_1_655_0",
            "3_0_102_1_656_0",
            "3_0_102_1_657_0",
            "3_0_102_1_658_0",
            "3_0_102_1_659_0",
            "3_0_102_1_682_0",
            "3_0_102_1_776_0",
            "3_0_102_1_788_0",
            "3_0_102_1_800_0",
            "3_0_102_1_812_0",
            "3_0_102_1_824_0",
            "3_0_102_1_836_0",
            "3_0_102_1_848_0",
            "3_0_102_1_860_0",
            "3_0_102_1_872_0",
            "3_0_102_1_884_0",
            "3_0_102_1_896_0",
            "3_0_102_1_908_0",
            "3_0_102_1_920_0",
            "3_0_103_1_652_0",
            "3_0_103_1_653_0",
            "3_0_103_1_654_0",
            "3_0_103_1_655_0",
            "3_0_103_1_656_0",
            "3_0_103_1_657_0",
            "3_0_103_1_658_0",
            "3_0_103_1_659_0",
            "3_0_103_1_682_0",
            "3_0_103_1_776_0",
            "3_0_103_1_788_0",
            "3_0_103_1_800_0",
            "3_0_103_1_812_0",
            "3_0_103_1_824_0",
            "3_0_103_1_836_0",
            "3_0_103_1_848_0",
            "3_0_103_1_860_0",
            "3_0_103_1_872_0",
            "3_0_103_1_884_0",
            "3_0_103_1_896_0",
            "3_0_103_1_908_0",
            "3_0_103_1_920_0",
            "3_0_103_1_932_0",
            "3_0_104_1_652_0",
            "3_0_104_1_653_0",
            "3_0_104_1_654_0",
            "3_0_104_1_655_0",
            "3_0_104_1_656_0",
            "3_0_104_1_657_0",
            "3_0_104_1_658_0",
            "3_0_104_1_659_0",
            "3_0_104_1_682_0",
            "3_0_104_1_776_0",
            "3_0_104_1_788_0",
            "3_0_104_1_800_0",
            "3_0_104_1_812_0",
            "3_0_104_1_824_0",
            "3_0_104_1_836_0",
            "3_0_104_1_848_0",
            "3_0_104_1_860_0",
            "3_0_104_1_872_0",
            "3_0_104_1_884_0",
            "3_0_104_1_896_0",
            "3_0_104_1_908_0",
            "3_0_104_1_920_0",
            "3_0_104_1_932_0",
            "3_0_105_1_652_0",
            "3_0_105_1_653_0",
            "3_0_105_1_654_0",
            "3_0_105_1_655_0",
            "3_0_105_1_656_0",
            "3_0_105_1_657_0",
            "3_0_105_1_658_0",
            "3_0_105_1_659_0",
            "3_0_105_1_682_0",
            "3_0_105_1_776_0",
            "3_0_105_1_788_0",
            "3_0_105_1_800_0",
            "3_0_105_1_812_0",
            "3_0_105_1_824_0",
            "3_0_105_1_836_0",
            "3_0_105_1_848_0",
            "3_0_105_1_860_0",
            "3_0_105_1_872_0",
            "3_0_105_1_884_0",
            "3_0_105_1_896_0",
            "3_0_105_1_908_0",
            "3_0_105_1_920_0",
            "3_0_105_1_932_0",
            "3_0_105_1_944_0",
            "3_0_106_1_652_0",
            "3_0_106_1_653_0",
            "3_0_106_1_654_0",
            "3_0_106_1_655_0",
            "3_0_106_1_656_0",
            "3_0_106_1_657_0",
            "3_0_106_1_658_0",
            "3_0_106_1_659_0",
            "3_0_106_1_682_0",
            "3_0_106_1_776_0",
            "3_0_106_1_788_0",
            "3_0_106_1_800_0",
            "3_0_106_1_812_0",
            "3_0_106_1_824_0",
            "3_0_106_1_836_0",
            "3_0_106_1_848_0",
            "3_0_106_1_860_0",
            "3_0_106_1_872_0",
            "3_0_106_1_884_0",
            "3_0_106_1_896_0",
            "3_0_106_1_908_0",
            "3_0_106_1_920_0",
            "3_0_106_1_932_0",
            "3_0_106_1_944_0",
            "3_0_107_1_652_0",
            "3_0_107_1_653_0",
            "3_0_107_1_654_0",
            "3_0_107_1_655_0",
            "3_0_107_1_656_0",
            "3_0_107_1_657_0",
            "3_0_107_1_658_0",
            "3_0_107_1_659_0",
            "3_0_107_1_682_0",
            "3_0_107_1_776_0",
            "3_0_107_1_788_0",
            "3_0_107_1_800_0",
            "3_0_107_1_812_0",
            "3_0_107_1_824_0",
            "3_0_107_1_836_0",
            "3_0_107_1_848_0",
            "3_0_107_1_860_0",
            "3_0_107_1_872_0",
            "3_0_107_1_884_0",
            "3_0_107_1_896_0",
            "3_0_107_1_908_0",
            "3_0_107_1_920_0",
            "3_0_107_1_932_0",
            "3_0_107_1_944_0",
            "3_0_108_1_652_0",
            "3_0_108_1_653_0",
            "3_0_108_1_654_0",
            "3_0_108_1_655_0",
            "3_0_108_1_656_0",
            "3_0_108_1_657_0",
            "3_0_108_1_658_0",
            "3_0_108_1_659_0",
            "3_0_108_1_682_0",
            "3_0_108_1_776_0",
            "3_0_108_1_788_0",
            "3_0_108_1_800_0",
            "3_0_108_1_812_0",
            "3_0_108_1_824_0",
            "3_0_108_1_836_0",
            "3_0_108_1_848_0",
            "3_0_108_1_860_0",
            "3_0_108_1_872_0",
            "3_0_108_1_884_0",
            "3_0_108_1_896_0",
            "3_0_108_1_908_0",
            "3_0_108_1_920_0",
            "3_0_108_1_932_0",
            "3_0_108_1_944_0",
            "3_0_109_1_652_0",
            "3_0_109_1_653_0",
            "3_0_109_1_654_0",
            "3_0_109_1_655_0",
            "3_0_109_1_656_0",
            "3_0_109_1_657_0",
            "3_0_109_1_658_0",
            "3_0_109_1_659_0",
            "3_0_109_1_682_0",
            "3_0_109_1_776_0",
            "3_0_109_1_788_0",
            "3_0_109_1_800_0",
            "3_0_109_1_812_0",
            "3_0_109_1_824_0",
            "3_0_109_1_836_0",
            "3_0_109_1_848_0",
            "3_0_109_1_860_0",
            "3_0_109_1_872_0",
            "3_0_109_1_884_0",
            "3_0_109_1_896_0",
            "3_0_109_1_908_0",
            "3_0_109_1_920_0",
            "3_0_109_1_932_0",
            "3_0_109_1_944_0",
            "3_0_109_1_956_0",
            "3_0_110_1_652_0",
            "3_0_110_1_653_0",
            "3_0_110_1_654_0",
            "3_0_110_1_655_0",
            "3_0_110_1_656_0",
            "3_0_110_1_657_0",
            "3_0_110_1_658_0",
            "3_0_110_1_659_0",
            "3_0_110_1_682_0",
            "3_0_110_1_776_0",
            "3_0_110_1_788_0",
            "3_0_110_1_800_0",
            "3_0_110_1_812_0",
            "3_0_110_1_824_0",
            "3_0_110_1_836_0",
            "3_0_110_1_848_0",
            "3_0_110_1_860_0",
            "3_0_110_1_872_0",
            "3_0_110_1_884_0",
            "3_0_110_1_896_0",
            "3_0_110_1_908_0",
            "3_0_110_1_920_0",
            "3_0_110_1_932_0",
            "3_0_110_1_944_0",
            "3_0_110_1_956_0",
            "3_0_111_1_652_0",
            "3_0_111_1_653_0",
            "3_0_111_1_654_0",
            "3_0_111_1_655_0",
            "3_0_111_1_656_0",
            "3_0_111_1_657_0",
            "3_0_111_1_658_0",
            "3_0_111_1_659_0",
            "3_0_111_1_682_0",
            "3_0_111_1_776_0",
            "3_0_111_1_788_0",
            "3_0_111_1_800_0",
            "3_0_111_1_812_0",
            "3_0_111_1_824_0",
            "3_0_111_1_836_0",
            "3_0_111_1_848_0",
            "3_0_111_1_860_0",
            "3_0_111_1_872_0",
            "3_0_111_1_884_0",
            "3_0_111_1_896_0",
            "3_0_111_1_908_0",
            "3_0_111_1_920_0",
            "3_0_111_1_932_0",
            "3_0_111_1_944_0",
            "3_0_112_1_652_0",
            "3_0_112_1_653_0",
            "3_0_112_1_654_0",
            "3_0_112_1_655_0",
            "3_0_112_1_656_0",
            "3_0_112_1_657_0",
            "3_0_112_1_658_0",
            "3_0_112_1_659_0",
            "3_0_112_1_682_0",
            "3_0_112_1_776_0",
            "3_0_112_1_788_0",
            "3_0_112_1_800_0",
            "3_0_112_1_812_0",
            "3_0_112_1_824_0",
            "3_0_112_1_836_0",
            "3_0_112_1_848_0",
            "3_0_112_1_860_0",
            "3_0_112_1_872_0",
            "3_0_112_1_884_0",
            "3_0_112_1_896_0",
            "3_0_112_1_908_0",
            "3_0_112_1_920_0",
            "3_0_112_1_932_0",
            "3_0_112_1_944_0",
            "4_0_101_1_652_0",
            "4_0_101_1_653_0",
            "4_0_101_1_654_0",
            "4_0_101_1_655_0",
            "4_0_101_1_656_0",
            "4_0_101_1_657_0",
            "4_0_101_1_658_0",
            "4_0_101_1_659_0",
            "4_0_101_1_682_0",
            "4_0_101_1_776_0",
            "4_0_101_1_788_0",
            "4_0_101_1_800_0",
            "4_0_101_1_812_0",
            "4_0_101_1_824_0",
            "4_0_101_1_836_0",
            "4_0_101_1_848_0",
            "4_0_101_1_860_0",
            "4_0_101_1_872_0",
            "4_0_101_1_884_0",
            "4_0_101_1_896_0",
            "4_0_101_1_908_0",
            "4_0_101_1_920_0",
            "4_0_101_1_932_0",
            "4_0_101_1_944_0",
            "4_0_102_1_652_0",
            "4_0_102_1_653_0",
            "4_0_102_1_654_0",
            "4_0_102_1_655_0",
            "4_0_102_1_656_0",
            "4_0_102_1_657_0",
            "4_0_102_1_658_0",
            "4_0_102_1_659_0",
            "4_0_102_1_682_0",
            "4_0_102_1_776_0",
            "4_0_102_1_788_0",
            "4_0_102_1_800_0",
            "4_0_102_1_812_0",
            "4_0_102_1_824_0",
            "4_0_102_1_836_0",
            "4_0_102_1_848_0",
            "4_0_102_1_860_0",
            "4_0_102_1_872_0",
            "4_0_102_1_884_0",
            "4_0_102_1_896_0",
            "4_0_102_1_908_0",
            "4_0_102_1_920_0",
            "4_0_102_1_932_0",
            "4_0_102_1_944_0",
            "4_0_103_1_652_0",
            "4_0_103_1_653_0",
            "4_0_103_1_654_0",
            "4_0_103_1_655_0",
            "4_0_103_1_656_0",
            "4_0_103_1_657_0",
            "4_0_103_1_658_0",
            "4_0_103_1_659_0",
            "4_0_103_1_682_0",
            "4_0_103_1_776_0",
            "4_0_103_1_788_0",
            "4_0_103_1_800_0",
            "4_0_103_1_812_0",
            "4_0_103_1_824_0",
            "4_0_103_1_836_0",
            "4_0_103_1_848_0",
            "4_0_103_1_860_0",
            "4_0_103_1_872_0",
            "4_0_103_1_884_0",
            "4_0_103_1_896_0",
            "4_0_103_1_908_0",
            "4_0_103_1_920_0",
            "4_0_103_1_932_0",
            "4_0_104_1_652_0",
            "4_0_104_1_653_0",
            "4_0_104_1_654_0",
            "4_0_104_1_655_0",
            "4_0_104_1_656_0",
            "4_0_104_1_657_0",
            "4_0_104_1_658_0",
            "4_0_104_1_659_0",
            "4_0_104_1_682_0",
            "4_0_104_1_776_0",
            "4_0_104_1_788_0",
            "4_0_104_1_800_0",
            "4_0_104_1_812_0",
            "4_0_104_1_824_0",
            "4_0_104_1_836_0",
            "4_0_104_1_848_0",
            "4_0_104_1_860_0",
            "4_0_104_1_872_0",
            "4_0_104_1_884_0",
            "4_0_104_1_896_0",
            "4_0_104_1_908_0",
            "4_0_104_1_920_0",
            "4_0_104_1_932_0",
            "56_0_701_2_600_0",
            "56_0_701_2_601_0",
            "56_0_701_2_602_0",
            "56_0_723_2_602_0",
            "56_0_725_1_600_0",
            "56_0_725_1_601_0",
            "6_0_261_2_623_0",
            "6_0_261_2_626_0",
            "6_0_261_2_662_0",
            "7_0_104_1_647_0",
            "7_0_104_1_650_0",
            "7_0_104_2_601_0",
            "7_0_104_2_602_0",
            "7_0_105_1_647_0",
            "7_0_105_1_650_0",
            "7_0_105_2_601_0",
            "7_0_105_2_602_0",
            "7_0_106_1_647_0",
            "7_0_106_1_650_0",
            "7_0_106_2_601_0",
            "7_0_106_2_602_0",
            "7_0_265_1_647_0",
            "7_0_265_1_650_0",
            "7_0_265_2_601_0",
            "7_0_265_2_602_0",
            "7_0_266_1_647_0",
            "7_0_266_1_650_0",
            "7_0_266_2_601_0",
            "7_0_266_2_602_0",
            "7_0_267_1_647_0",
            "7_0_267_1_650_0",
            "7_0_267_2_601_0",
            "7_0_267_2_602_0",
            "7_0_268_1_647_0",
            "7_0_268_1_650_0",
            "7_0_268_2_601_0",
            "7_0_268_2_602_0",
            "7_0_300_1_647_0",
            "7_0_300_1_650_0",
            "7_0_300_2_601_0",
            "7_0_300_2_602_0",
            "7_0_301_1_647_0",
            "7_0_301_1_650_0",
            "7_0_301_2_601_0",
            "7_0_301_2_602_0",
            "7_0_302_1_647_0",
            "7_0_302_1_650_0",
            "7_0_302_2_601_0",
            "7_0_302_2_602_0",
            "7_0_306_1_647_0",
            "7_0_306_1_650_0",
            "7_0_306_2_601_0",
            "7_0_306_2_602_0",
            "7_0_308_1_625_0",
            "7_0_308_1_632_0",
            "7_0_308_1_633_0",
            "7_0_308_1_634_0",
            "7_0_308_1_651_0",
            "7_0_308_1_652_0",
            "7_0_308_1_653_0",
            "7_0_308_2_718_0",
            "7_0_308_2_769_0",
            "7_0_308_2_920_0",
            "7_0_310_1_625_0",
            "7_0_310_1_632_0",
            "7_0_310_1_633_0",
            "7_0_310_1_634_0",
            "7_0_310_1_651_0",
            "7_0_310_1_652_0",
            "7_0_310_1_653_0",
            "7_0_310_2_718_0",
            "7_0_310_2_769_0",
            "7_0_310_2_920_0",
            "7_0_311_1_625_0",
            "7_0_311_1_632_0",
            "7_0_311_1_633_0",
            "7_0_311_1_634_0",
            "7_0_311_1_651_0",
            "7_0_311_1_652_0",
            "7_0_311_1_653_0",
            "7_0_311_2_718_0",
            "7_0_311_2_769_0",
            "7_0_311_2_920_0",
            "7_0_312_1_625_0",
            "7_0_312_1_632_0",
            "7_0_312_1_633_0",
            "7_0_312_1_634_0",
            "7_0_312_1_651_0",
            "7_0_312_1_652_0",
            "7_0_312_1_653_0",
            "7_0_312_2_718_0",
            "7_0_312_2_769_0",
            "7_0_312_2_920_0",
            "7_0_317_1_647_0",
            "7_0_317_1_650_0",
            "7_0_317_2_601_0",
            "7_0_317_2_602_0",
            "7_0_319_1_635_0",
            "8_0_168_1_635_0",
            "8_0_169_1_635_0",
            "8_0_172_2_623_0",
            "8_0_172_2_626_0",
            "8_0_172_2_662_0",
            "9_0_171_1_647_0",
            "9_0_171_1_650_0",
            "9_0_171_2_601_0",
            "9_0_171_2_602_0",
            "9_0_172_1_647_0",
            "9_0_172_1_650_0",
            "9_0_172_2_601_0",
            "9_0_172_2_602_0",
            "9_0_173_1_647_0",
            "9_0_173_1_650_0",
            "9_0_173_2_601_0",
            "9_0_173_2_602_0",
            "9_0_174_1_647_0",
            "9_0_174_1_650_0",
            "9_0_174_2_601_0",
            "9_0_174_2_602_0",
            "9_0_180_1_635_0",
            "9_0_181_1_635_0",
        ]
        for idx, file_path in enumerate(glob.glob(data_dir)):
            if file_path.split("\\")[-1][:-26] in files_303_304:
                import shutil
                new_dir = shutil.move(file_path, new_data_dir)




# 测试代码 main 函数
def main():
    """
    # ------------------------------
    # 加载 A3 楼 301,302 csv 数据文件，并转换为一个整体文件
    # ------------------------------
    # 实例化
    data_loader = DataLoad(building_floor = "A3F3")
    # 数据保存路径
    if sys.platform == "win32":
        tf_data_path = f"{data_loader.data_dir}\\tf_data\\all_df.csv"
    else:
        tf_data_path = f"{data_loader.data_dir}/tf_data/all_df.csv"
    # 数据解析
    if not os.path.exists(tf_data_path):
        df = data_loader.load_A3F3_transformer_data()
        df.to_csv(tf_data_path, index=False)
    else:
        df = pd.read_csv(tf_data_path)
        logger.info(f"df: \n{df.head()}")
        logger.info(f"df shape: {df.shape}")
        logger.info(f"df columns: \n{df.columns}")
        logger.info(f"df 301_load: \n{df['301_load'].head()}")
        logger.info(f"df 302_load: \n{df['302_load'].head()}")
        logger.info(f"'301_load' in df.columns: {'301_load' in df.columns}")
        logger.info(f"'302_load' in df.columns: {'302_load' in df.columns}")
    # ------------------------------
    # 加载 A1 楼 201,202,203,204 csv 数据文件，并转换为一个整体文件
    # ------------------------------
    # 实例化
    data_loader = DataLoad(building_floor = "A1F2")
    # 数据保存路径
    if sys.platform == "win32":
        tf_data_path = f"{data_loader.data_dir}\\tf_data\\all_df.csv"
    else:
        tf_data_path = f"{data_loader.data_dir}/tf_data/all_df.csv"
    # 数据解析
    if not os.path.exists(tf_data_path):
        df = data_loader.load_A1F2_transformer_data() 
        df.to_csv(tf_data_path, index=False)
    else:
        df = pd.read_csv(tf_data_path)
        logger.info(f"df: \n{df.head()}")
        logger.info(f"df shape: {df.shape}")
        logger.info(f"df columns: \n{df.columns}")
        logger.info(f"df 201_load: \n{df['201_load'].head()}")
        logger.info(f"df 202_load: \n{df['202_load'].head()}")
        logger.info(f"df 203_load: \n{df['203_load'].head()}")
        logger.info(f"df 204_load: \n{df['204_load'].head()}")
        logger.info(f"'201_load' in df.columns: {'201_load' in df.columns}")
        logger.info(f"'202_load' in df.columns: {'202_load' in df.columns}")
        logger.info(f"'203_load' in df.columns: {'203_load' in df.columns}")
        logger.info(f"'204_load' in df.columns: {'204_load' in df.columns}")
    # ------------------------------
    # 加载 A3 201,202,203,204 csv 数据文件，并转换为一个整体文件
    # ------------------------------
    # 实例化
    data_loader = DataLoad(building_floor = "A3F2")
    # 数据保存路径
    if sys.platform == "win32":
        tf_data_path = f"{data_loader.data_dir}\\tf_data\\all_df.csv"
    else:
        tf_data_path = f"{data_loader.data_dir}/tf_data/all_df.csv"
    # 数据解析
    if not os.path.exists(tf_data_path):
        df = data_loader.load_A3F2_transformer_data() 
        df.to_csv(tf_data_path, index=False) 
    else:
        df = pd.read_csv(tf_data_path)
        logger.info(f"df: \n{df.head()}")
        logger.info(f"df shape: {df.shape}")
        logger.info(f"df columns: \n{df.columns}")
        logger.info(f"df 201_load: \n{df['201_load'].head()}")
        logger.info(f"df 202_load: \n{df['202_load'].head()}")
        logger.info(f"df 203_load: \n{df['203_load'].head()}")
        logger.info(f"df 204_load: \n{df['204_load'].head()}")
        logger.info(f"'201_load' in df.columns: {'201_load' in df.columns}")
        logger.info(f"'202_load' in df.columns: {'202_load' in df.columns}")
        logger.info(f"'203_load' in df.columns: {'203_load' in df.columns}")
        logger.info(f"'204_load' in df.columns: {'204_load' in df.columns}")
    """
    # ------------------------------
    # 
    # ------------------------------
    # 实例化
    # data_loader = DataLoad(building_floor = "A1F2")
    # data_loader.get_A1F2_files()
    # 实例化
    data_loader = DataLoad(building_floor = "A3F3")
    # data_loader.get_A3F3_files()
    data_loader.move_unused_files()
    

if __name__ == "__main__":
    main()
