import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy
import math
import warnings
warnings.filterwarnings("ignore")
import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.metrics import (
    r2_score,  # R2
    mean_squared_error,  # MSE
    root_mean_squared_error,  # RMSE
    mean_absolute_error,  # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import StandardScaler

from utils.log_util import logger


class ModelMainClass:

    def __init__(self, args) -> None:
        # ------------------------------
        # 特征工程
        # ------------------------------
        self.n_lags = len(args.lags)  # 特征滞后数个数(1,2,...)
        self.date_type = None
        # ------------------------------
        # 模型预测
        # ------------------------------
        # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
        self.horizon = args.predict_days * args.n_per_day
        # ------------------------------
        # 数据窗口
        # ------------------------------
        # 测试滑动窗口数量, >=1, 1: 单个窗口
        self.n_windows = args.history_days - (args.window_days - 1)
        # 测试窗口数据长度(训练+测试)
        self.window_len = args.window_days * args.n_per_day if self.n_windows > 1 else args.history_days * args.n_per_day
        # ------------------------------
        # 数据划分时间
        # ------------------------------
        now = datetime.datetime(2025, 5, 19, 0, 0, 0)  # 模型预测的日期时间
        self.now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)  # 时间序列当前时刻
        self.start_time = self.now_time.replace(hour=0) - datetime.timedelta(days=self.history_days)  # 时间序列历史数据开始时刻
        self.future_time = self.now_time + datetime.timedelta(days=self.predict_days)  # 时间序列未来结束时刻
        self.before_days = -args.history_days
        self.after_days = args.predict_days
        # ------------------------------
        # 模型参数
        # ------------------------------
        self.model_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "max_bin": 31,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
        }
        # ------------------------------
        # log
        # ------------------------------
        self.log_prefix = ""
        logger.info(f"\nhistory data range: {self.start_time}~{self.now_time}")
        logger.info(f"\npredict data range: {self.now_time}~{self.future_time}")

    def _get_data(self):
        df_target = pd.read_csv(self.args.data_)
        df_date = None
        df_weather = None
        self.input_data = {
            "df_power": df_target,
            "df_date": df_date,
            "df_weather": df_weather,
            "df_date_future": df_date,
            "df_weather_future": df_weather,
        }

    def _preprocess_data(self, raw_df: pd.DataFrame, column_name: str, new_column_name: str):
        # 复制 future data
        df = copy.deepcopy(raw_df)
        # 数据处理
        if df is not None:
            # 转换时间戳类型
            df[new_column_name] = pd.to_datetime(df[column_name])
            # 去除重复时间戳
            df.drop_duplicates(subset=new_column_name, keep="last", inplace=True, ignore_index=True)

        return df

    def process_history_data(self):
        """
        处理历史数据
        """
        # 数据预处理
        if self.input_data["df_power"] is not None:
            df_power = self._preprocess_data(self.input_data["df_power"], self.power_timestamp_feat, "timeStamp")
            logger.info(f"{self.log_prefix} df_power shape: {df_power.shape}")
        else:
            df_power = None
            logger.info(f"{self.log_prefix} df_power is None.")
        
        if self.input_data["df_date"] is not None:
            df_date = self._preprocess_data(self.input_data["df_date"], self.date_timestamp_feat, "timeStamp")
            logger.info(f"{self.log_prefix} df_date shape: {df_date.shape}")
        else:
            df_date = None
            logger.info(f"{self.log_prefix} df_date is None.")
        
        if self.input_data["df_weather"] is not None:
            df_weather = self._preprocess_data(self.input_data["df_weather"], self.weather_timestamp_feat, "timeStamp")
            logger.info(f"{self.log_prefix} df_weather shape: {df_weather.shape}")
        else:
            df_weather = None
            logger.info(f"{self.log_prefix} df_weather is None.")
        # 整理历史功率数据
        df_load = pd.DataFrame({"timeStamp": pd.date_range(self.start_time, self.now_time, freq=self.freq, inclusive="left")})
        for col in df_power.columns:
            if col not in [self.power_timestamp_feat, "timeStamp"]:
                df_load[col] = df_power[col].apply(lambda x: float(x))
                df_load[col] = df_load["timeStamp"].map(df_power.set_index("timeStamp")[col])
        logger.info(f"{self.log_prefix} df_load shape after map load: {df_load.shape}")
        # 删除含空值的行
        df_load.dropna(inplace=True, ignore_index=True)
        logger.info(f"{self.log_prefix} df_load shape after drop NA: {df_load.shape}")
        # 删除需求负荷小于 0 的样本
        df_load = df_load[df_load[self.target] > 0]
        logger.info(f"{self.log_prefix} df_load shape after data filter: {df_load.shape}")
        logger.info(f"{self.log_prefix} df_load has nan or not: \n{df_load.isna().any()}")  # 缺失值检查
        # 特征工程: 外生特征
        exogenous_features = [col for col in df_load.columns if col != "timeStamp" and col != self.target]
        logger.info(f"{self.log_prefix} df_load exogenous_features: {exogenous_features}")
        # 特征工程: 天气特征
        if df_weather is not None:
            df_load, weather_features = self.extend_weather_feature(df_load, df_weather)
            logger.info(f"{self.log_prefix} df_load shape after merge weather features: {df_load.shape} \ndf_load.columns: {df_load.columns}")
        else:
            weather_features = []
        # 特征工程: 时间特征
        df_load, datetime_features = self.extend_datetime_feature(df_load)
        logger.info(f"{self.log_prefix} df_load shape after merge datetime features: {df_load.shape} \ndf_load.columns: {df_load.columns}")
        # 特征工程: 日期特征
        if df_date is not None:
            df_load, date_features = self.extend_date_type_feature(df_load, df_date)
            logger.info(f"{self.log_prefix} df_load shape after merge date features: {df_load.shape} \ndf_load.columns: {df_load.columns}")
        else:
            date_features = []
        # 特征工程: 滞后特征
        if exogenous_features == [] or (not self.target_transform and self.pred_method != "multivariate-multip-step-recursive"):
            df_load, lag_features = self.extend_lag_feature_univariate(
                df=df_load,
                target=self.target, 
                lags=self.lags,
            )
        else:
            df_load, lag_features, target_features = self.extend_lag_feature_multivariate(
                df=df_load,
                exogenous_features=exogenous_features,
                target=self.target,
                n_lags=self.n_lags,
            )
        logger.info(f"{self.log_prefix} df_load shape after merge lag features: {df_load.shape} \ndf_load.columns: {df_load.columns}")
        # 缺失值处理
        df_load = df_load.interpolate()  # 缺失值插值填充
        df_load.dropna(inplace=True, ignore_index=True)  # 缺失值删除
        logger.info(f"{self.log_prefix} df_load shape after process NA: {df_load.shape}")
        # 特征排序
        if self.pred_method == "univariate-multip-step-recursive":
            predict_features = [col for col in lag_features if self.target in col] + weather_features + datetime_features + date_features
        else:
            predict_features = lag_features + weather_features + datetime_features + date_features
        logger.info(f"{self.log_prefix} predict_features: \n{predict_features}")
        
        if exogenous_features == [] or (not self.target_transform and self.pred_method != "multivariate-multip-step-recursive"):
            df_load = df_load[["timeStamp"] + predict_features + [self.target]]
        else:
            df_load = df_load[["timeStamp"] + predict_features + target_features]
        logger.info(f"{self.log_prefix} df_load shape after feature engineering: {df_load.shape}")
        logger.info(f"{self.log_prefix} df_load.head() after feature engineering: \n{df_load.head()}")
        logger.info(f"{self.log_prefix} df_load.tail() after feature engineering: \n{df_load.tail()}")
        # 工作日
        # df_load_workday = copy.deepcopy(df_load.loc[(
        #     (df_load["timeStamp"] < "2025-05-17 21:00:00") |
        #     ((df_load["timeStamp"] > "2025-05-19 07:00:00") & (df_load["timeStamp"] < "2025-05-24 21:00:00")) |
        #     ((df_load["timeStamp"] > "2025-05-26 07:00:00") & (df_load["timeStamp"] < "2025-05-30 16:00:00")) |
        #     ((df_load["timeStamp"] > "2025-06-03 07:00:00") & (df_load["timeStamp"] < "2025-06-07 21:00:00")) |
        #     (df_load["timeStamp"] > "2025-06-09 07:00:00"))
        #     , 
        # ])
        df_load_workday = copy.deepcopy(df_load[df_load["datetime_weekday"] < 5])
        # df_load_workday = copy.deepcopy(df_load[df_load[self.target] > 400])
        logger.info(f"{self.log_prefix} df_load_workday.shape: {df_load_workday.shape}")
        # 非工作日
        # df_load_offday = copy.deepcopy(df_load.loc[(
        #     ((df_load["timeStamp"] >= "2025-05-17 21:00:00") & (df_load["timeStamp"] <= "2025-05-19 07:00:00")) |
        #     ((df_load["timeStamp"] >= "2025-05-24 21:00:00") & (df_load["timeStamp"] <= "2025-05-26 07:00:00")) |
        #     ((df_load["timeStamp"] >= "2025-05-30 16:00:00") & (df_load["timeStamp"] <= "2025-06-03 07:00:00")) |
        #     ((df_load["timeStamp"] >= "2025-06-07 21:00:00") & (df_load["timeStamp"] <= "2025-06-09 07:00:00")))
        #     ,
        # ])
        df_load_offday = copy.deepcopy(df_load[df_load["datetime_weekday"] >= 5])
        # df_load_offday = copy.deepcopy(df_load[df_load[self.target] < 400])
        logger.info(f"{self.log_prefix} df_load_offday.shape: {df_load_offday.shape}")
        
        if self.date_type is not None:
            # workday 预测特征、目标变量分割
            data_X_workday = df_load_workday[predict_features]
            if exogenous_features == [] or not self.target_transform:
                data_Y_workday = df_load_workday[self.target]
            else:
                data_Y_workday = df_load_workday[target_features]
            if isinstance(data_Y_workday, pd.DataFrame):
                data_Y_workday.columns = [col.replace("(t+1)", "") for col in data_Y_workday.columns]
            # offday 预测特征、目标变量分割
            data_X_offday = df_load_offday[predict_features]
            if exogenous_features == [] or not self.target_transform:
                data_Y_offday = df_load_offday[self.target]
            else:
                data_Y_offday = df_load_offday[target_features]
            if isinstance(data_Y_offday, pd.DataFrame):
                data_Y_offday.columns = [col.replace("(t+1)", "") for col in data_Y_offday.columns]
            return data_X_workday, data_Y_workday, data_X_offday, data_Y_offday
        else:
            # 预测特征、目标变量分割
            data_X = df_load[predict_features]
            if exogenous_features == [] or (not self.target_transform and self.pred_method != "multivariate-multip-step-recursive"):
                data_Y = df_load[self.target]
            else:
                data_Y = df_load[target_features]
            if isinstance(data_Y, pd.DataFrame):
                data_Y.columns = [col.replace("(t+1)", "") for col in data_Y.columns]

            return data_X, data_Y

    def process_future_data(self):
        """
        处理未来数据
        """
        # 数据预处理
        if self.input_data["df_date_future"] is not None:
            df_date_future = self._preprocess_data(self.input_data["df_date_future"], self.date_timestamp_feat, "timeStamp")
            logger.info(f"{self.log_prefix} df_date_future shape: {df_date_future.shape}")
        else:
            df_date_future = None
        
        if self.input_data["df_weather_future"] is not None:
            df_weather_future = self._preprocess_data(self.input_data["df_weather_future"], self.weather_timestamp_feat, "timeStamp")
            logger.info(f"{self.log_prefix} df_weather_future shape: {df_weather_future.shape}")
        else:
            df_weather_future = None
        # 创建 DataFrame 并添加 timeStamp 列
        df_future = pd.DataFrame({
            "timeStamp": pd.date_range(
                pd.to_datetime(self.now_time).replace(minute=0, second=0, microsecond=0),
                self.future_time, 
                freq=self.freq,
                inclusive="left"
            )
        })
        logger.info(f"{self.log_prefix} df_future: \n{df_future}")
        # 特征工程: 外生特征
        exogenous_features = [col for col in df_future.columns if col != "timeStamp" and col != self.target]
        logger.info(f"{self.log_prefix} df_load exogenous_features: {exogenous_features}")
        # 特征工程
        df_future, datetime_features = self.extend_datetime_feature(df_future)
        logger.info(f"{self.log_prefix} df_future shape after merge datetime features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
        # 日期时间特征
        if df_date_future is not None:
            df_future, date_features = self.extend_date_type_feature(df_future, df_date_future)
        else:
            date_features = []
        logger.info(f"{self.log_prefix} df_future shape after merge date features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
        # 环境特征
        if df_weather_future is not None:
            df_future, weather_features = self.extend_future_weather_feature(df_future, df_weather_future)
        else:
            weather_features = []
        logger.info(f"{self.log_prefix} df_future shape after merge weather features: {df_future.shape} \ndf_future.columns: {df_future.columns}")
        # 滞后特征
        lag_features = []
        logger.info(f"{self.log_prefix} df_load lag_features: {lag_features}")
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_future.dropna(inplace=True, ignore_index=True)
        logger.info(f"{self.log_prefix} df_future shape after interpolate and dropna: {df_future.shape}")
        # 特征列表
        future_feature_list = lag_features + weather_features + datetime_features + date_features
        logger.info(f"{self.log_prefix} future_feature_list: \n{future_feature_list}")
        df_future_copy = df_future.copy()
        # 工作日
        # df_future_workday = copy.deepcopy(df_future.loc[(
        #     (df_future["timeStamp"] < "2025-05-17 21:00:00") |
        #     ((df_future["timeStamp"] > "2025-05-19 07:00:00") & (df_future["timeStamp"] < "2025-05-24 21:00:00")) |
        #     ((df_future["timeStamp"] > "2025-05-26 07:00:00") & (df_future["timeStamp"] < "2025-05-30 16:00:00")) |
        #     ((df_future["timeStamp"] > "2025-06-03 07:00:00") & (df_future["timeStamp"] < "2025-06-07 21:00:00")) |
        #     (df_future["timeStamp"] > "2025-06-09 07:00:00"))
        #     , 
        # ])
        df_future_workday = copy.deepcopy(df_future[df_future["datetime_weekday"] < 5])
        # df_future_workday = copy.deepcopy(df_future[df_future[self.target] > 400])
        logger.info(f"{self.log_prefix} df_future_workday.shape: {df_future_workday.shape}")
        # 非工作日
        # df_future_offday = copy.deepcopy(df_future.loc[(
        #     ((df_future["timeStamp"] >= "2025-05-17 21:00:00") & (df_future["timeStamp"] <= "2025-05-19 07:00:00")) |
        #     ((df_future["timeStamp"] >= "2025-05-24 21:00:00") & (df_future["timeStamp"] <= "2025-05-26 07:00:00")) |
        #     ((df_future["timeStamp"] >= "2025-05-30 16:00:00") & (df_future["timeStamp"] <= "2025-06-03 07:00:00")) |
        #     ((df_future["timeStamp"] >= "2025-06-07 21:00:00") & (df_future["timeStamp"] <= "2025-06-09 07:00:00")))
        #     ,
        # ])
        df_future_offday = copy.deepcopy(df_future[df_future["datetime_weekday"] >= 5])
        # df_load_offday = copy.deepcopy(df_future[df_future[self.target] < 400])
        logger.info(f"{self.log_prefix} df_future_offday.shape: {df_future_offday.shape}")
        if self.date_type is not None:
            # workday 截取未来数据
            df_future_workday.set_index("timeStamp", inplace=True)
            data_X_future_workday = df_future_workday.iloc[-self.horizon:, ]
            data_X_future_workday = df_future_workday.loc[:, future_feature_list]
            logger.info(f"{self.log_prefix} data_X_future_workday: \n{data_X_future_workday.head()} \ndata_X_future_workday.columns: \n{data_X_future_workday.columns}")
            # offday 截取未来数据
            df_future_offday.set_index("timeStamp", inplace=True)
            data_X_future_offday = df_future_offday.iloc[-self.horizon:, ]
            data_X_future_offday = df_future_offday.loc[:, future_feature_list]
            logger.info(f"{self.log_prefix} data_X_future_offday: \n{data_X_future_offday.head()} \ndata_X_future_offday.columns: \n{data_X_future_offday.columns}")
            return data_X_future_workday, data_X_future_offday, df_future_copy
        else:
            # 截取未来数据
            df_future.set_index("timeStamp", inplace=True)
            data_X_future = df_future.iloc[-self.horizon:, ]
            data_X_future = df_future.loc[:, future_feature_list]
            logger.info(f"{self.log_prefix} data_X_future: \n{data_X_future.head()} \ndata_X_future.columns: \n{data_X_future.columns}")

            return data_X_future, df_future_copy

    def extend_datetime_feature(self, df: pd.DataFrame):
        """
        增加时间特征
        """
        df["datetime_minute"] = df["timeStamp"].apply(lambda x: x.minute)
        df["datetime_hour"] = df["timeStamp"].apply(lambda x: x.hour)
        df["datetime_day"] = df["timeStamp"].apply(lambda x: x.day)
        df["datetime_weekday"] = df["timeStamp"].apply(lambda x: x.weekday())
        df["datetime_week"] = df["timeStamp"].apply(lambda x: x.week)
        df["datetime_day_of_week"] = df["timeStamp"].apply(lambda x: x.dayofweek)
        df["datetime_week_of_year"] = df["timeStamp"].apply(lambda x: x.weekofyear)
        df["datetime_month"] = df["timeStamp"].apply(lambda x: x.month)
        df["datetime_days_in_month"] = df["timeStamp"].apply(lambda x: x.daysinmonth)
        df["datetime_quarter"] = df["timeStamp"].apply(lambda x: x.quarter)
        df["datetime_day_of_year"] = df["timeStamp"].apply(lambda x: x.dayofyear)
        df["datetime_year"] = df["timeStamp"].apply(lambda x: x.year)
        datetime_features = [
            "datetime_minute",
            "datetime_hour",
            "datetime_day",
            "datetime_weekday",
            "datetime_week",
            "datetime_day_of_week",
            "datetime_week_of_year",
            "datetime_month",
            "datetime_days_in_month",
            "datetime_quarter",
            "datetime_day_of_year",
            "datetime_year",
        ]

        return df, datetime_features

    def extend_date_type_feature(self, df: pd.DataFrame, df_date: pd.DataFrame):
        """
        增加日期类型特征：
        1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
        """
        # data map
        df["date"] = df["timeStamp"].apply(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        df["date_type"] = df["date"].map(df_date.set_index("timeStamp")["date_type"])
        # date features
        date_features = ["date_type"]

        return df, date_features

    def extend_weather_feature(self, df: pd.DataFrame, df_weather: pd.DataFrame):
        """
        处理天气特征
        """
        # 特征筛选
        weather_features_raw = [
            "rt_ssr",
            "rt_ws10",
            "rt_tt2",
            "rt_dt",
            "rt_ps",
            "rt_rain",
        ]
        df_weather = df_weather[["timeStamp"] + weather_features_raw]
        # 删除含空值的行
        df_weather.dropna(inplace=True, ignore_index=True)
        # 将除了timeStamp的列转为float类型
        for col in weather_features_raw:
            df_weather[col] = df_weather[col].apply(lambda x: float(x))
        # 计算相对湿度
        df_weather["cal_rh"] = np.nan
        for i in df_weather.index:
            if (
                df_weather.loc[i, "rt_tt2"] is not np.nan
                and df_weather.loc[i, "rt_dt"] is not np.nan
            ):
                # 通过温度和露点温度计算相对湿度
                temp = (
                    math.exp(
                        17.2693
                        * (df_weather.loc[i, "rt_dt"] - 273.15)
                        / (df_weather.loc[i, "rt_dt"] - 35.86)
                    )
                    / math.exp(
                        17.2693
                        * (df_weather.loc[i, "rt_tt2"] - 273.15)
                        / (df_weather.loc[i, "rt_tt2"] - 35.86)
                    )
                    * 100
                )
                temp = max(min(temp, 100), 0)
                df_weather.loc[i, "cal_rh"] = temp
            else:
                rt_tt2 = df_weather.loc[i, "rt_tt2"]
                rt_dt = df_weather.loc[i, "rt_dt"]
                logger.info(f"{self.log_prefix} rt_tt2 is {rt_tt2}, rt_dt is {rt_dt}")
        # 特征排序
        weather_features = [
            "rt_ssr",  # 太阳总辐射
            "rt_ws10",  # 10m 风速
            "rt_tt2",  # 2M 气温
            "cal_rh",  # 相对湿度
            "rt_ps",  # 气压
            "rt_rain",  # 降雨量
        ]
        df_weather = df_weather[["timeStamp"] + weather_features]
        # 合并功率数据和天气数据
        df = pd.merge(df, df_weather, on="timeStamp", how="left")
        # 插值填充缺失值
        df = df.interpolate()
        df.dropna(inplace=True, ignore_index=True)

        return df, weather_features

    def extend_future_weather_feature(self, df_future, df_weather_future):
        """
        未来天气数据特征构造

        Args:
            df_future (_type_): _description_
            df_weather_future (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 筛选天气预测数据
        pred_weather_features = [
            "pred_ssrd",
            "pred_ws10",
            "pred_tt2",
            "pred_rh",
            "pred_ps",
            "pred_rain",
        ]
        df_weather_future = df_weather_future[["timeStamp"] + pred_weather_features]
        # 删除含空值的行
        df_weather_future.dropna(inplace=True, ignore_index=True)
        # 数据类型转换
        for col in pred_weather_features:
            df_weather_future[col] = df_weather_future[col].apply(lambda x: float(x))
        # 将预测天气数据整理到预测df中
        df_future["rt_ssr"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ssrd"]
        )
        df_future["rt_ws10"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ws10"]
        )
        df_future["rt_tt2"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_tt2"]
        )
        df_future["cal_rh"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_rh"]
        )
        df_future["rt_ps"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ps"]
        )
        df_future["rt_rain"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_rain"]
        )

        weather_features = ["rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"]

        return df_future, weather_features

    def extend_lag_feature_univariate(self, df: pd.DataFrame, target: str, lags: List):
        """
        添加滞后特征
        """
        df_lags = df.copy()
        # lag features building
        for lag in lags:
            df_lags[f"{target}_{lag}"] = df_lags[target].shift(lag)
        df_lags.dropna(inplace=True)
        # features
        lag_features = [f"{target}_{lag}" for lag in lags]

        return df_lags, lag_features
    
    @staticmethod
    def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int, return_Xy: bool = False):
        """
        Time delay embedding
        Time series for supervised learning

        Args:
            series: time series as pd.Series
            n_lags: number of past values to used as explanatory variables
            horizon: how many values to forecast
            return_Xy: whether to return the lags split from future observations

        Return: pd.DataFrame with reconstructed time series
        """
        assert isinstance(series, pd.Series)
        # series name
        if series.name is None:
            name = 'Series'
        else:
            name = series.name
        # create features
        n_lags_iter = list(range(n_lags, -horizon, -1))
        df_list = [series.shift(i) for i in n_lags_iter]
        df = pd.concat(df_list, axis=1).dropna()
        # features rename
        df.columns = [
            f'{name}(t-{j - 1})' if j > 0 else f'{name}(t+{np.abs(j) + 1})'
            for j in n_lags_iter
        ]
        # df.columns = [re.sub('t-0', 't', x) for x in df.columns]
        # 返回 pandas.Dataframe
        if not return_Xy:
            return df
        # future features
        is_future = df.columns.str.contains('\\+')
        # feature split
        X = df.iloc[:, ~is_future]
        Y = df.iloc[:, is_future]
        if Y.shape[1] == 1:
            Y = Y.iloc[:, 0]

        return X, Y
    
    def extend_lag_feature_multivariate(self, df: pd.DataFrame, exogenous_features: List, target: str, n_lags: int):
        """
        添加滞后特征
        """
        # 将 date 作为索引
        df.set_index("timeStamp", inplace=True)
        # delay embedding: lagged features
        lagged_features_ds = []
        for col in exogenous_features + [target]:
            col_df = self.time_delay_embedding(
                series=df[col], 
                n_lags=n_lags, 
                horizon = 1
            )
            lagged_features_ds.append(col_df)
            df = df.drop(columns=[col])
        lagged_features_df = pd.concat(lagged_features_ds, axis=1).dropna()
        lagged_features_df = lagged_features_df.reset_index()
        # 滞后特征提取
        lag_features = [
            col for col in lagged_features_df.columns 
            if col.__contains__("(t-") or col.__contains__(r"(t)")
        ]
        # 目标特征提取
        target_features = [
            col for col in lagged_features_df.columns 
            if col.__contains__("(t+")
        ]
        # 数据合并
        df = df.reset_index()
        df = lagged_features_df.merge(df, on = "timeStamp", how = "left")
        
        return df, lag_features, target_features
    
    @staticmethod
    def univariate_directly_forecast(model, X_test):
        """
        模型预测
        """
        if len(X_test) > 0:
            Y_pred = model.predict(X_test)
            return Y_pred
        else:
            Y_pred = []
            logger.info(f"X_future length is 0!")
            return Y_pred

    def univariate_recursive_forecast(self, model, X_train, Y_train, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        # last 96xday's steps true targets
        pred_history = list(Y_train.iloc[-int(max(lags)):-1][self.target].values)
        # initial features
        training_feature_list = [col for col in X_train.columns if col != "ds"]
        current_features_df = X_train[training_feature_list].copy()
        # forecast collection
        Y_pred = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            if not isinstance(next_pred[0], np.ndarray):
                pred_history.append(next_pred[0])
            else:
                pred_history.append(next_pred[0][-1])
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            for i in lags:
                if i > len(pred_history): break
                if not self.target_transform:
                    new_row_df[f"{self.target}_{i}"] = pred_history[-i]
                else:
                    new_row_df[f"{self.target}(t-{i-1})"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # 收集预测结果
            Y_pred.append(next_pred[0])

        return Y_pred

    def multivariate_recursive_forecast(self, model, X_train, Y_train, future, lags, steps, scaler_features=None):
        """
        递归多步预测
        """
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        # last 96xday's steps true targets
        pred_history = list(Y_train.iloc[-int(max(lags)):].values)
        # initial features
        training_feature_list = [col for col in X_train.columns if col != "ds"]
        current_features_df = X_train[training_feature_list].copy()
        # forecast collection
        Y_pred = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(current_features_df.iloc[-1:])
            else:
                current_features = current_features_df.iloc[-1:].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            pred_history.append(next_pred[0])
            pred_history_list = np.array(pred_history[-self.n_lags:]).T.flatten().tolist()
            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            new_row_df.iloc[:, 0:(Y_train.shape[1]*self.n_lags)] = pred_history_list
            # 更新 current_features_df
            current_features_df = pd.concat([current_features_df, new_row_df], axis=0, ignore_index=True)
            # TODO 收集预测结果
            # Y_pred.append(next_pred[0][-1])
            Y_pred.append(next_pred[0])

        return Y_pred

    def _evaluate_split_index(self, window: int):
        """
        数据分割索引构建
        """
        test_end    = -1         + (-self.horizon) * (window - 1)
        test_start  = test_end   + (-self.horizon) + 1
        train_end   = test_start
        train_start = test_end   + (-self.window_len) + 1

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, window: int):
        """
        训练、测试数据集分割
        """
        # 数据分割指标
        train_start, train_end, \
        test_start, test_end = self._evaluate_split_index(window)
        # 数据分割
        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        if test_end == -1:
            X_test = data_X.iloc[test_start:]
            Y_test = data_Y.iloc[test_start:]
        else:
            X_test = data_X.iloc[test_start:(test_end+1)]
            Y_test = data_Y.iloc[test_start:(test_end+1)]

        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def _evaluate_score(Y_test, Y_pred, window: int):
        """
        模型评估
        """
        # 计算模型的性能指标
        if Y_test.mean() == 0:
            Y_test = Y_test.apply(lambda x: x + 0.01)
        eval_scores = {
            "R2": r2_score(Y_test, Y_pred),
            "mse": mean_squared_error(Y_test, Y_pred),
            "rmse": root_mean_squared_error(Y_test, Y_pred),
            "mae": mean_absolute_error(Y_test, Y_pred),
            "mape": mean_absolute_percentage_error(Y_test, Y_pred),
            "mape accuracy": 1 - mean_absolute_percentage_error(Y_test, Y_pred) 
        }
        eval_scores = pd.DataFrame(eval_scores, index=[window])
        
        return eval_scores

    def _evaluate_result(self, Y_test, Y_pred, window: int):
        """
        测试集预测数据
        """ 
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window)
        # 训练结果数据收集
        cv_plot_df_window = pd.DataFrame()
        cv_timestamp_df = pd.DataFrame({"ds": pd.date_range(start=self.start_time, end=self.now_time, freq=self.freq, inclusive="left")})
        if test_end == -1:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{''}")
        else:
            cv_plot_df_window["ds"] = cv_timestamp_df[test_start:(test_end+1)]
            logger.info(f"split indexes:: train_start:train_end: {train_start}:{train_end}, test_start:test_end: {test_start}:{test_end+1}")
        cv_plot_df_window["train_start"] = [cv_timestamp_df["ds"].values[train_start]] * len(Y_pred)
        cv_plot_df_window["cutoff"] = [cv_timestamp_df["ds"].values[test_start]] * len(Y_pred)
        cv_plot_df_window["test_end"] = [cv_timestamp_df["ds"].values[test_end]] * len(Y_pred)
        cv_plot_df_window["Y_trues"] = Y_test
        cv_plot_df_window["Y_preds"] = Y_pred
        
        return cv_plot_df_window
    
    def _window_test(self, X_train, Y_train, X_test):
        """
        模型训练
        """
        # 特征列表
        feature_list = X_train.columns
        # 训练集、测试集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        # 归一化/标准化
        if self.scale:
            scaler_features_test = StandardScaler()
            X_train[feature_list] = scaler_features_test.fit_transform(X_train[feature_list])
            X_test[feature_list] = scaler_features_test.transform(X_test[feature_list])
        else:
            scaler_features_test = None
        # 模型训练
        if Y_train.shape[1] == 1:
            model = lgb.LGBMRegressor(**self.model_params)
            model.fit(X_train, Y_train)
        else:
            model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_params))
            # model = RegressorChain(lgb.LGBMRegressor(**self.model_params))
            model.fit(X_train, Y_train)
        # 模型预测
        if self.pred_method == "multip-step-directly":
            Y_pred = self.univariate_directly_forecast(
                model = model,
                X_test = X_test,
            )
        elif self.pred_method == "univariate-multip-step-recursive":
            Y_pred = self.univariate_recursive_forecast(
                model = model,
                X_train = X_train_df, 
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.lags,
                steps = min(self.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        elif self.pred_method == "multivariate-multip-step-recursive":
            Y_pred = self.multivariate_recursive_forecast(
                model = model,
                X_train = X_train_df, 
                Y_train = Y_train_df,
                future = X_test_df,
                lags = self.lags,
                steps = min(self.horizon, len(X_test_df)),
                scaler_features = scaler_features_test,
            )
        
        return Y_pred

    def test(self, data_X: pd.DataFrame, data_Y):
        """
        交叉验证
        """
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        for window in range(1, self.n_windows + 1):
            logger.info(f"{'-' * 40}")
            logger.info(f"training window: {window}...")
            logger.info(f"{'-' * 40}")
            # 数据分割: 训练集、测试集
            X_train, Y_train, X_test, Y_test = self._evaluate_split(data_X, data_Y, window)
            logger.info(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
            if X_train.shape[0] == 0:
                break
            # 目标特征处理
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            if self.target_transform and not self.target_transform_predict and self.pred_method != "multivariate-multip-step-recursive":
                Y_train = Y_train.iloc[:, -1:]
                Y_train.columns = [self.target]
            # 模型测试
            Y_pred = self._window_test(X_train, Y_train, X_test)
            # 多变量, 单变量直接多步预测: 目标特征转换
            if self.target_transform:
                # test
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_test_ups_output = Y_test.iloc[:, 0:1].values
                Y_test            = Y_test.iloc[:, -1:].values
                Y_test = Y_test * Y_test_ups_output - Y_test_ups_output
                if not self.target_transform_predict:
                    # pred
                    Y_pred = np.array(Y_pred)
                    logger.info(f"Y_pred.shape[0]: {len(Y_pred.shape)}")
                    if len(Y_pred.shape) == 1:
                        Y_pred = Y_pred.reshape(-1, 1)
                    else:
                        Y_pred = pd.DataFrame(Y_pred, columns=data_Y.columns)
                        Y_pred = Y_pred.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_test_ups_output - Y_test_ups_output
                elif self.target_transform_predict:
                    # pred
                    Y_preds = pd.DataFrame(Y_pred, columns=data_Y.columns)
                    Y_pred_ups_output = Y_preds.iloc[:, 0:1].values
                    Y_pred            = Y_preds.iloc[:, -1:].values
                    Y_pred = Y_pred * Y_pred_ups_output - Y_pred_ups_output
            elif not self.target_transform and self.pred_method == "multivariate-multip-step-recursive":
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_preds = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_test = Y_test.iloc[:, -1:].values
                Y_pred = Y_preds.iloc[:, -1:].values
            else:
                Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
                Y_test = Y_test.iloc[:, -1:].values
            # 测试集评价指标
            eval_scores = self._evaluate_score(Y_test, Y_pred, window)
            test_scores_df = pd.concat([test_scores_df, eval_scores], axis = 0)
            # 测试集预测数据
            cv_plot_df_window = self._evaluate_result(Y_test, Y_pred, window)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis = 0)
        logger.info(f"cross validation test_scores_df: \n{test_scores_df}")
        logger.info(f"cross validation cv_plot_df: \n{cv_plot_df}")
        # 模型评价指标数据处理
        test_scores_df = test_scores_df.mean()
        test_scores_df = test_scores_df.to_frame().T.reset_index(drop = True, inplace = False)
        logger.info(f"cross validation average scores: \n{test_scores_df}")
        
        return test_scores_df, cv_plot_df
    
    def train(self, data_X: pd.DataFrame, data_Y):
        """
        模型训练
        """
        # 特征列表
        feature_list = data_X.columns
        # 所有训练数据
        final_X_train = data_X.copy()
        final_Y_train = data_Y.copy()
        # 归一化/标准化
        if self.scale:
            scaler_features = StandardScaler()
            final_X_train[feature_list] = scaler_features.fit_transform(final_X_train[feature_list])
        else:
            scaler_features = None
        # 模型训练
        if isinstance(final_Y_train, pd.Series):
            final_model = lgb.LGBMRegressor(**self.model_params)
            final_model.fit(final_X_train, final_Y_train)
        else:
            final_model = MultiOutputRegressor(lgb.LGBMRegressor(**self.model_params))
            # final_model = RegressorChain(lgb.LGBMRegressor(**self.model_params))
            final_model.fit(final_X_train, final_Y_train)

        return final_model, scaler_features

    def forecast(self, model, data_X, data_Y, data_X_future, data_future=None, scaler_features=None):
        """
        模型预测
        """
        # 模型预测
        if len(data_X_future) > 0:
            # directly multi-step forecast
            if self.pred_method == "multip-step-directly":
                if scaler_features is not None:
                    X_test_future = data_X_future.copy()
                    X_test_future = scaler_features.transform(X_test_future)
                else:
                    X_test_future = data_X_future
                Y_pred = self.univariate_directly_forecast(
                    model = model,
                    X_test = X_test_future
                )
            # recursive multi-step forecast
            elif self.pred_method == "univariate-multip-step-recursive":
                Y_pred = self.univariate_recursive_forecast(
                    model = model,
                    X_train = data_X, 
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.lags,
                    steps = min(self.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            elif self.pred_method == "multivariate-multip-step-recursive":
                Y_pred = self.multivariate_recursive_forecast(
                    model = model,
                    X_train = data_X, 
                    Y_train = data_Y,
                    future = data_X_future,
                    lags = self.lags,
                    steps = min(self.horizon, len(data_X_future)),
                    scaler_features = scaler_features,
                )
            # 多变量, 单变量直接多步预测: 目标变量转换
            if self.target_transform:
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_future_ups_output = Y_pred_df.iloc[:, 0:1].values
                Y_pred = Y_pred_df.iloc[:, -1:].values
                Y_pred = Y_pred * Y_future_ups_output - Y_future_ups_output
            elif not self.target_transform and self.pred_method == "multivariate-multip-step-recursive":
                Y_pred_df = pd.DataFrame(Y_pred, columns=data_Y.columns)
                Y_pred = Y_pred_df.iloc[:, -1:].values
            # 预测结果收集
            data_X_future[self.target] = Y_pred
        else:
            data_X_future[self.target] = np.nan
        logger.info(f"{self.log_prefix} data_X_future after forecast: \n{data_X_future.head()} \ndata_X_future length after forecast: {len(data_X_future)}")
        # 输出结果处理
        # data_X_future.dropna(inplace=True, ignore_index=False)
        # logger.info(f"{self.log_prefix} data_X_future after dropna: \n{data_X_future.head()} \ndata_X_future length after dropna: {len(data_X_future)}")
        if self.date_type is not None:
            return data_X_future
        else:
            data_future = pd.merge(data_future, data_X_future, how="outer")
            return data_future

    def process_output(self, df_future):
        df_future.reset_index(drop=False, inplace=True)
        for i in range(len(df_future)):
            df_future.loc[i, "id"] = (f"{self.node_id}_{self.out_system_id}_{df_future.loc[i, 'timeStamp'].strftime('%Y%m%d%H%M%S')}")
            df_future.loc[i, "node_id"] = self.node_id
            df_future.loc[i, "system_id"] = self.out_system_id
            df_future.loc[i, "count_data_time"] = df_future.loc[i, "timeStamp"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            df_future.loc[i, "predict_value"] = str(df_future.loc[i, self.target])
        df_future = df_future[[
            "id",
            "node_id",
            "system_id",
            "count_data_time",
            "predict_value",
        ]]
        logger.info(f"df_future: \n{df_future}")

        return df_future

    def run(self, input_data: Dict, model_cfgs: Dict):
        """
        实际负荷预测
        """
        # ------------------------------
        # 历史数据预处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model history data preprocessing...")
        logger.info(f"{80*'='}")
        if self.date_type is not None:
            data_X_workday, data_Y_workday, data_X_offday, data_Y_offday = self.process_history_data()
        else:
            data_X, data_Y = self.process_history_data()
        logger.info(f"Model history data preprocessing over...")
        # ------------------------------
        # 模型测试
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model testing...")
        logger.info(f"{80*'='}")
        if self.date_type is not None:
            test_scores_df, cv_plot_df = self.test(data_X_workday, data_Y_workday)
            test_scores_df, cv_plot_df = self.test(data_X_offday, data_Y_offday)
        else:
            test_scores_df, cv_plot_df = self.test(data_X, data_Y)
        logger.info(f"Model testing over...")
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model training...")
        logger.info(f"{80*'='}")
        if self.date_type is not None:
            model_workday, scaler_features_workday = self.train(data_X_workday, data_Y_workday)
            model_offday, scaler_features_offday = self.train(data_X_offday, data_Y_offday)
        else:
            model, scaler_features = self.train(data_X, data_Y)
        logger.info(f"Model training over...")
        # ------------------------------
        # 未来数据处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model future data preprocessing...")
        logger.info(f"{80*'='}")
        logger.info(f"{self.log_prefix} future data process...")
        if self.date_type is not None:
            data_X_future_workday, data_X_future_offday, df_future = self.process_future_data()
        else:
            data_X_future, data_future = self.process_future_data()
        logger.info(f"Model history data preprocessing over...")
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model Forecast...")
        logger.info(f"{80*'='}")
        if self.date_type is not None:
            pred_df_workday = self.forecast(
                model_workday,
                data_X_workday, data_Y_workday,
                data_X_future_workday, None,
                scaler_features_workday,
            )
            pred_df_offday = self.forecast(
                model_offday,
                data_X_offday, data_Y_offday,
                data_X_future_offday, None,
                scaler_features_offday
            )
            df_future = pd.merge(df_future, pred_df_workday, how="outer")
            df_future = pd.merge(df_future, pred_df_offday, how="outer")
        else:
            df_future = self.forecast(
                model, 
                data_X, data_Y, 
                data_X_future, data_future, 
                scaler_features
            )
        logger.info(f"Model forecasting result: \n{df_future}")
        logger.info(f"Model forecasting over...")
        # ------------------------------
        # 输出结果处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Forecast result processing...")
        logger.info(f"{80*'='}")
        df_future_output = self.process_output(df_future)
        # 模型输出
        return {"df_future": df_future_output}




# 测试代码 main 函数
def main():
    args = None
    model_ins = ModelMainClass(args)
    result = model_ins.run()

if __name__ == "__main__":
    main()
