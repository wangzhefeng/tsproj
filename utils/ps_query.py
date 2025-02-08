# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ps_query.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-11-14
# * Version     : 1.0.111417
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
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import datetime
from typing import Dict

import pandas as pd
import psycopg2
import pytz

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PostgresQuery:
    
    def __init__(self, 
                 conn_params: Dict, 
                 now_time: datetime.datetime,
                 history_days: int = 30,
                 future_days: int = 5):
        # ------------------------------
        # predict
        # ------------------------------
        # 历史时间戳
        self.start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)
        self.start_time_str = self.start_time.strftime("%Y/%m/%d %H:%M:%S")
        # 现在时间戳
        self.end_time = now_time
        self.end_time_str = now_time.strftime("%Y/%m/%d %H:%M:%S")
        # 未来时间戳
        self.future_time = now_time + datetime.timedelta(days=future_days)
        self.future_time_str = self.future_time.strftime("%Y/%m/%d %H:%M:%S")
        
        print(f"predict data start_time: {self.start_time_str}")
        print(f"predict data end_time: {self.end_time_str}")
        print(f"predict data future_time: {self.future_time_str}")
        # ------------------------------
        # strategy
        # ------------------------------
        self.start_time_stra = (now_time + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        self.end_time_stra = (now_time + datetime.timedelta(days=1)).replace(hour=23, minute=48, second=0, microsecond=0)
        self.start_time_str_stra = self.start_time_stra.strftime("%Y-%m-%d %H:%M:%S")
        self.end_time_str_stra = self.end_time_stra.strftime("%Y-%m-%d %H:%M:%S")
        self.start_month_str_stra = self.start_time_stra.replace(day=1).strftime("%Y-%m-%d")
        
        print(f"strategy data start_time: {self.start_time_str_stra}")
        print(f"strategy data end_time: {self.end_time_str_stra}")
        print(f"strategy data start_month: {self.start_month_str_stra}")
        
        # 创建数据库连接
        self.conn = psycopg2.connect(**conn_params)
        # 创建 SQL cursor 对象
        self.cur = self.conn.cursor()
    
    # ------------------------------
    # predict
    # ------------------------------
    def query_history(self, table_name: str, ts_col_name: str = "date", 
                      metric: str = None, node_id: str=None, point_sn: str=None): 
        # 查询历史数据
        if metric is None:
            select_query = f"""
                SELECT * 
                FROM {table_name} 
                WHERE {ts_col_name} BETWEEN '{self.start_time_str}' AND '{self.end_time_str}'
                ORDER BY {ts_col_name};
                """
        elif metric == "power":
            # 查询历史有功功率
            select_query = f"""
                SELECT count_data_time, h_total_use
                FROM {table_name} 
                WHERE 
                    node_id = '{node_id}' AND 
                    point_sn = '{point_sn}' AND 
                    count_data_time BETWEEN '{self.start_time_str}' AND '{self.end_time_str}' 
                ORDER BY count_data_time;"""
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns)

    def query_future(self, table_name: str, ts_col_name: str = "date"):
        # 查询历史到现在的工作日历
        select_query = f"""
            SELECT * 
            FROM {table_name}
            WHERE {ts_col_name} BETWEEN '{self.end_time_str}' AND '{self.future_time_str}'
            ORDER BY {ts_col_name};
            """
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns)    

    # ------------------------------
    # strategy
    # ------------------------------
    def query_weather_data_stra(self, table_name: str): 
        # 查询气象数据
        select_query = f"""
            SELECT * 
            FROM {table_name} 
            WHERE ts BETWEEN '{self.start_time_str_stra}' AND '{self.end_time_str_stra}'
            ORDER BY ts;
            """
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns)

    def query_date_data_stra(self, table_name: str):
        # 查询工作日历数据
        select_query = f"""
            SELECT * 
            FROM {table_name} 
            WHERE date BETWEEN '{self.start_time_str_stra}' AND '{self.end_time_str_stra}'
            ORDER BY date;
            """
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns)
    
    def query_power_pred_stra(self, table_name, node_id, system_id):
        # 查询预测的需求负荷
        select_query = f"""
            SELECT * 
            FROM {table_name}
            WHERE 
                node_id = '{node_id}' AND 
                system_id = '{system_id}' AND 
                count_data_time BETWEEN '{self.start_time_str_stra}' AND '{self.end_time_str_stra}' 
            ORDER BY count_data_time;
            """
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns)

    def query_power_price_stra(self, table_name: str, node_id: str=None, system_id: str=None):
        # 查询当月电价
        select_query = f"""
            SELECT * 
            FROM {table_name}
            WHERE 
                node_id = '{node_id}' AND 
                system_id = '{system_id}' AND 
                effective_date BETWEEN '{self.start_month_str_stra}' AND '{self.start_month_str_stra}'
            ORDER BY s_time;
            """
        self.cur.execute(select_query)
        # 获取查询结果
        rows = self.cur.fetchall()
        # 列名称
        columns = [desc[0] for desc in self.cur.description]
        # 将数据转换为DataFrame
        self.df = pd.DataFrame(rows, columns=columns) 

    def save_history(self, local_data_path: str="./dataset/ashichuang/", 
                     node_name: str = "", data_name: str = ""):
        data_path = f"{data_name}_{self.start_time.strftime("%Y%m%d")}_to_{self.end_time.strftime("%Y%m%d")}.csv"
        data_path = f"{node_name}_" + data_path if node_name != "" else data_path
        self.df.to_csv(
            os.path.join(local_data_path, data_path), 
            encoding='utf_8_sig', 
            index=False
        )

    def save_future(self, local_data_path: str="./dataset/ashichuang/", 
                    node_name: str = "", data_name: str = ""):
        data_path = f"{data_name}_{self.end_time.strftime("%Y%m%d")}_to_{self.future_time.strftime("%Y%m%d")}.csv"
        data_path = f"{node_name}_" + data_path if node_name != "" else data_path
        self.df.to_csv(
            os.path.join(local_data_path, data_path), 
            encoding='utf_8_sig', 
            index=False
        )

    def save_strategy(self, local_data_path: str="./dataset/ashichuang/", node_name: str = "", data_name: str = ""):
        data_path = f"{data_name}_{self.start_time_stra.strftime("%Y%m%d")}.csv"
        data_path = f"{node_name}_" + data_path if node_name != "" else data_path
        self.df.to_csv(
            os.path.join(local_data_path, data_path), 
            encoding='utf_8_sig', 
            index=False
        )

    def close(self):
        # 关闭游标和连接
        self.cur.close()
        self.conn.close()




# 测试代码 main 函数
def main():
    # 气象数据库连接参数
    conn_params_wh = {
        "host": "47.122.59.9",  # 测试环境数据库
        # "host": "host.docker.internal",
        "dbname": "sensor_data_raw",
        "user": "postgres",
        "password": "QTparking123456@",
    }
    # 数据库连接参数
    conn_params = {
        "host": "47.100.89.197",  # 测试环境
        # "host": "47.122.59.9",  # 演示环境
        # "host": "47.101.39.138",  # 全国环境
        # "host": "host.docker.internal",
        "dbname": "damao_vpp_01_00_00_resource_slice_test", # 测试环境数据库
        # "dbname": "vpp_01_00_00_resourceType", # 演示环境数据库
        # "dbname": "vpp_01_04_00_resourceType", # 现场环境数据库
        "user": "postgres",
        "password": "QTparking123456@",
    }

    # SQL查询表名与字段
    table_name_w = "szx_weather"  # 气象数据
    calendar_table = 'demand_calendar'  # 工作日历
    table_name = 'iot_ts_kv_metering_device_96'  # 15分钟切片数据表 
    table_name_2 = 'cfg_storage_energy_strategy'  # 电价数据表
    table_name_1 = 'ai_load_forecasting'  # 预测模型输出表

    # now time
    now_time = datetime.datetime(2024, 11, 20, 00, 00, 0)
    history_days = 30
    predict_days = 5
    # system params
    node = ["asc1", "asc2"]
    node_id = ["f79237fc44855884911d8136ba431f5c", "f7a388e48987a8003245d4c7028fed70"]
    node_name = ["阿石创新材料公司储能组1", "阿石创新材料公司储能组2"]
    system_id = ["nengyuanzongbiao","chuneng"]
    point_sn = {
        "f79237fc44855884911d8136ba431f5c": ["g1-load", "g1-gkb-load"],
        "f7a388e48987a8003245d4c7028fed70": ["g2-load", "g2-gkb-load"],
    }

    # query
    pq = PostgresQuery(conn_params = conn_params, now_time=now_time, history_days=history_days, future_days=predict_days)
    pq_wh = PostgresQuery(conn_params=conn_params_wh, now_time=now_time, history_days=history_days, future_days=predict_days)
    # ------------------------------
    # 预测模型数据查询 
    # ------------------------------
    # 工作日历数据下载
    # ---------------
    # df_date history
    pq.query_history(table_name=calendar_table, ts_col_name="date")
    pq.save_history(local_data_path="./dataset/ashichuang/pred/", data_name="df_date")
    # df_date future
    pq.query_future(table_name=calendar_table, ts_col_name="date")
    pq.save_future(local_data_path="./dataset/ashichuang/pred/", data_name="df_date")
    # 负荷数据
    # ---------------
    # 储能组 1
    # PCS 总有功功率
    pq.query_history(table_name=table_name, ts_col_name="date", metric="power", node_id = node_id[0], point_sn=point_sn[node_id[0]][0])
    pq.save_history(local_data_path="./dataset/ashichuang/pred/", node_name=node[0], data_name="df_es_1")
    # 组1关口表有功功率
    pq.query_history(table_name=table_name, ts_col_name="date", metric="power", node_id = node_id[0], point_sn=point_sn[node_id[0]][1])
    pq.save_history(local_data_path="./dataset/ashichuang/pred/", node_name=node[0], data_name="df_gate")
    # 储能组 2
    # PCS 总有功功率
    pq.query_history(table_name=table_name, ts_col_name="date", metric="power", node_id = node_id[1], point_sn=point_sn[node_id[1]][0])
    pq.save_history(local_data_path="./dataset/ashichuang/pred/", node_name=node[1], data_name="df_es_1")
    # 组2关口表有功功率
    pq.query_history(table_name=table_name, ts_col_name="date", metric="power", node_id = node_id[1], point_sn=point_sn[node_id[1]][1])
    pq.save_history(local_data_path="./dataset/ashichuang/pred/", node_name=node[1], data_name="df_gate") 
    # 气象数据
    # ---------------
    # df_weather history
    pq_wh.query_history(table_name=table_name_w, ts_col_name="ts")
    pq_wh.save_history(local_data_path="./dataset/ashichuang/pred/", data_name="df_weather")
    # df_weather future
    pq_wh.query_future(table_name=table_name_w, ts_col_name="ts")
    pq_wh.save_future(local_data_path="./dataset/ashichuang/pred/", data_name="df_weather")
    # ------------------------------
    # 储能调度优化数据查询
    # ------------------------------
    # 天气数据
    pq_wh.query_weather_data_stra(table_name=table_name_w)
    pq_wh.save_strategy(local_data_path="./dataset/ashichuang/stra/", data_name="df_weather")
    # 工作日历数据
    pq.query_date_data_stra(table_name=calendar_table)
    pq.save_strategy(local_data_path="./dataset/ashichuang/stra/", data_name="df_date")
    # 储能组1: 负荷预测数据
    pq.query_power_pred_stra(table_name=table_name_1, node_id=node_id[0], system_id=system_id[0])
    pq.save_strategy(local_data_path="./dataset/ashichuang/stra/", node_name=node[0], data_name="df_power")
    # 储能组2: 负荷预测数据
    pq.query_power_pred_stra(table_name=table_name_1, node_id=node_id[1], system_id=system_id[0])
    pq.save_strategy(local_data_path="./dataset/ashichuang/stra/", node_name=node[1], data_name="df_power")
    # 储能组1：电价数据
    pq.query_power_price_stra(table_name=table_name_2, node_id=node_id[0], system_id=system_id[0])
    pq.save_strategy(local_data_path="./dataset/ashichuang/stra/", node_name=node[0], data_name="df_price")
    # 储能组2：电价数据
    pq.query_power_price_stra(table_name=table_name_2, node_id=node_id[1], system_id=system_id[0])
    pq.save_strategy(local_data_path="./dataset/ashichuang/stra/", node_name=node[1], data_name="df_price")
    # ------------------------------
    # 关闭连接
    # ------------------------------
    pq.close()
    pq_wh.close()

if __name__ == "__main__":
    main()
