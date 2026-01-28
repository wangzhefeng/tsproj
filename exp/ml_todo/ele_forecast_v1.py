import copy
import math
import random
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    r2_score,  # R2
    mean_squared_error,  # MSE
    root_mean_squared_error,  # RMSE
    mean_absolute_error,  # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import StandardScaler

# from model import BaseModelMainClass
from utils.log_util import logger


class ModelMainClass:#(BaseModelMainClass):

    def __init__(self, project, model, node, args: Dict) -> None:
        self.project = project
        self.model = model
        self.node = node
        self.args = args
        self.log_prefix = f"project: {project}, model: {model}, node: {node}::"

    def _preprocess_data(
        self, raw_df: pd.DataFrame, column_name: str, new_column_name: str
    ):
        # copy
        df = copy.deepcopy(raw_df)
        # 转换时间戳类型
        df[new_column_name] = pd.to_datetime(df[column_name])
        # 去除重复时间戳
        df.drop_duplicates(
            subset=new_column_name, keep="last", inplace=True, ignore_index=True
        )

        return df

    def process_history_data(self):
        """
        处理历史数据
        """
        # 数据预处理
        df_power = self._preprocess_data(
            self.input_data["df_power"], "count_data_time", "timeStamp"
        )
        df_date = self._preprocess_data(self.input_data["df_date"], "date", "timeStamp")
        df_weather = self._preprocess_data(
            self.input_data["df_weather"], "ts", "timeStamp"
        )
        # 整理历史功率数据
        df_load = pd.DataFrame(
            {"timeStamp": pd.date_range(self.start_time, self.now_time, freq=self.freq)}
        )
        df_load[self.target] = df_load["timeStamp"].map(
            df_power.set_index("timeStamp")["h_total_use"]
        )  # 将原始数据映射到时间戳完整的 df_load 中
        df_load[self.target] = df_load[self.target].apply(
            lambda x: float(x)
        )  # 功率数据转换为浮点数
        logger.info(f"{self.log_prefix} df_load length after map load: {len(df_load)}")
        df_load.dropna(inplace=True, ignore_index=True)  # 删除含空值的行
        logger.info(f"{self.log_prefix} df_load length after drop NA: {len(df_load)}")
        df_load = df_load[df_load[self.target] > 0]  # 如果需求负荷小于 0，删除
        logger.info(
            f"{self.log_prefix} df_load length after data filter: {len(df_load)}"
        )
        logger.info(
            f"{self.log_prefix} df_load has nan or not: \n{df_load.isna().any()}"
        )  # 缺失值检查
        # 特征工程
        df_load, weather_features = self.extend_weather_feature(df_load, df_weather)
        logger.info(
            f"{self.log_prefix} df_load length after merge weather features: {len(df_load)}"
        )
        df_load, datetime_features = self.extend_datetime_stamp_feature(df_load)
        logger.info(
            f"{self.log_prefix} df_load length after merge datetime features: {len(df_load)}"
        )
        df_load, date_features = self.extend_date_type_feature(df_load, df_date)
        logger.info(
            f"{self.log_prefix} df_load length after merge date features: {len(df_load)}"
        )
        df_load, lag_features = self.extend_lag_feature(df_load, lags=self.lags)
        logger.info(
            f"{self.log_prefix} df_load length after merge lag features: {len(df_load)}"
        )
        # 特征排序
        training_feature_list = (
            lag_features + weather_features + datetime_features + date_features
        )
        df_load = df_load[["timeStamp"] + training_feature_list + [self.target]]
        logger.info(
            f"{self.log_prefix} training_feature_list: \n{training_feature_list}"
        )
        logger.info(
            f"{self.log_prefix} df_load length after feature engineering: {len(df_load)}"
        )
        logger.info(
            f"{self.log_prefix} df_load.head() after feature engineering: \n{df_load.head()}"
        )
        logger.info(
            f"{self.log_prefix} df_load.tail() after feature engineering: \n{df_load.tail()}"
        )
        '''
        # workday 样本
        df_load_workday = copy.deepcopy(df_load[df_load["date_type"] == 1])
        logger.info(
            f"{self.log_prefix} length of df_load_workday: {len(df_load_workday)}"
        )
        # offday 样本、特征分割
        df_load_offday = copy.deepcopy(df_load[df_load["date_type"] > 1])
        logger.info(
            f"{self.log_prefix} length of df_load_offday: {len(df_load_offday)}"
        )

        return (df_load_workday, df_load_offday, training_feature_list)
        '''

        return (df_load, training_feature_list)

    def process_future_data(self):
        """
        处理未来数据
        """
        # 数据预处理
        df_date_future = self._preprocess_data(
            self.input_data["df_date_future"], "date", "timeStamp"
        )
        df_weather_future = self._preprocess_data(
            self.input_data["df_weather_future"], "ts", "timeStamp"
        )
        # 创建 DataFrame 并添加 timeStamp 列
        df_future = pd.DataFrame(
            {
                "timeStamp": pd.date_range(
                    pd.to_datetime(self.now_time).replace(minute=0, second=0, microsecond=0), 
                    self.future_time, freq=self.freq
                )
            }
        )
        # 特征工程
        df_future, datetime_features = self.extend_datetime_stamp_feature(df_future)
        logger.info(
            f"{self.log_prefix} df_future length after merge datetime features: {len(df_future)}"
        )
        df_future, date_features = self.extend_date_type_feature(
            df_future, df_date_future
        )
        logger.info(
            f"{self.log_prefix} df_future length after merge date features: {len(df_future)}"
        )
        df_future, weather_features = self.extend_future_weather_feature(
            df_future, df_weather_future
        )
        logger.info(
            f"{self.log_prefix} df_future length after merge weather features: {len(df_future)}"
        )
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_future.dropna(inplace=True, ignore_index=True)
        logger.info(
            f"{self.log_prefix} df_future length after interpolate and dropna: {len(df_future)}"
        )
        # 特征列表
        future_feature_list = weather_features + datetime_features + date_features
        logger.info(f"{self.log_prefix} future_feature_list: \n{future_feature_list}")
        '''
        # 数据分割：工作日预测特征
        df_future_workday = copy.deepcopy(df_future[df_future["date_type"] == 1])
        logger.info(
            f"{self.log_prefix} length of df_future_workday: {len(df_future_workday)}"
        )
        # 数据分割：非工作日特征
        df_future_offday = copy.deepcopy(df_future[df_future["date_type"] > 1])
        logger.info(
            f"{self.log_prefix} length of df_future_offday: {len(df_future_offday)}"
        )

        return (df_future, df_future_workday, df_future_offday, future_feature_list)
        '''

        return (df_future, future_feature_list)

    def extend_datetime_stamp_feature(self, df: pd.DataFrame):
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

    def extend_weather_feature(self, df_load: pd.DataFrame, df_weather: pd.DataFrame):
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
        df_load = pd.merge(df_load, df_weather, on="timeStamp", how="left")
        # 插值填充缺失值
        df_load = df_load.interpolate()
        df_load.dropna(inplace=True, ignore_index=True)

        return df_load, weather_features

    def extend_lag_feature(self, df: pd.DataFrame, lags: List):
        """
        添加滞后特征
        """

        for lag in lags:
            df[f"lag_{lag}"] = df[self.target].shift(lag)
        df.dropna(inplace=True)

        lag_features = [f"lag_{lag}" for lag in lags]

        return df, lag_features

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

    def _recursive_forecast(
        self, model, history, future, lags, steps, scaler_features=None
    ):
        """
        递归多步预测
        """
        # last 96xday's steps true targets
        pred_history = list(history.iloc[-int(max(lags)) : -1][self.target].values)
        # initial features
        training_feature_list = [
            col for col in history.columns if col not in ["timeStamp", self.target]
        ]
        current_features_df = history[training_feature_list].copy()
        # forecast collection
        predictions = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(
                    current_features_df.iloc[-1:]
                )
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            pred_history.append(next_pred[0])

            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            for i in lags:
                if i > len(pred_history):
                    break
                new_row_df[f"lag_{i}"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat(
                [current_features_df, new_row_df],
                axis=0,
                ignore_index=True,
            )

            # 收集预测结果
            predictions.append(next_pred[0])

        return predictions

    def train(self, data_X, data_Y, lgbm_params):
        """
        模型训练
        """
        # 特征列表
        feature_list = data_X.columns
        # 训练集、测试集划分
        data_length = len(data_X)
        X_train = data_X.iloc[-data_length : -self.split_length]
        Y_train = data_Y.iloc[-data_length : -self.split_length]
        X_test = data_X.iloc[-self.split_length :]
        Y_test = data_Y.iloc[-self.split_length :]
        # 训练集、测试集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        Y_test_df = Y_test.copy()
        # ------------------------------
        # 模型测试
        # ------------------------------
        # 归一化/标准化
        if self.scale:
            scaler_features_test = StandardScaler()
            X_train[feature_list] = scaler_features_test.fit_transform(
                X_train[feature_list]
            )
            # X_test[feature_list] = scaler_features_test.transform(X_test[feature_list])
        else:
            scaler_features_test = None
        # 模型训练
        lgb_model = lgb.LGBMRegressor(**lgbm_params)
        lgb_model.fit(X_train, Y_train)
        # 模型预测
        if self.lags == []:
            Y_predicted = lgb_model.predict(X_test)
        else:
            Y_predicted = self._recursive_forecast(
                model=lgb_model,
                history=pd.concat([X_train_df, Y_train_df], axis=1),
                future=X_test_df,
                lags=self.lags,
                steps=self.split_length,
                scaler_features=scaler_features_test,
            )
        # logger.info(f"{self.log_prefix} Y_predicted: {Y_predicted} \nY_predicted length: {len(Y_predicted)}")
        # 模型评价
        # y_train_list = list(np.array(Y_train_df))
        # y_test_list = list(np.array(Y_test_df))
        # y_test_list = [None for _ in range(len(y_train_list) - len(y_test_list))] + y_test_list
        # y_pred_list = [None for _ in range(len(y_train_list) - len(Y_predicted))] + list(Y_predicted)
        # test_results = pd.DataFrame({
        #     "Y_train": y_train_list,
        #     "Y_test": y_test_list,
        #     "Y_pred": y_pred_list,
        # })
        # test_results.to_csv(f"./logs/test_results_{self.model}/test_results.csv", index=False, encoding="utf_8_sig")
        test_scores = {
            "R2": r2_score(Y_test_df, Y_predicted),
            "mse": mean_squared_error(Y_test_df, Y_predicted),
            "rmse": root_mean_squared_error(Y_test_df, Y_predicted),
            "mae": mean_absolute_error(Y_test_df, Y_predicted),
            "mape": mean_absolute_percentage_error(Y_test_df, Y_predicted),
            "accuracy": 1 - mean_absolute_percentage_error(Y_test_df, Y_predicted),
        }
        logger.info(f"{self.log_prefix} model test R2: {test_scores['R2']:.4f}")
        logger.info(f"{self.log_prefix} model test mse: {test_scores['mse']:.4f}")
        logger.info(f"{self.log_prefix} model test rmse: {test_scores['rmse']:.4f}")
        logger.info(f"{self.log_prefix} model test mae: {test_scores['mae']:.4f}")
        logger.info(f"{self.log_prefix} model test mape: {test_scores['mape']:.4f}")
        logger.info(f"{self.log_prefix} model test mape accuracy: {test_scores['accuracy']:.4f}")
        # ------------------------------
        # 最终模型
        # ------------------------------
        # 所有训练数据
        final_X_train = pd.concat([X_train_df, X_test_df], axis=0)
        final_Y_train = pd.concat([Y_train_df, Y_test_df], axis=0)
        # 归一化/标准化
        if self.scale:
            scaler_features = StandardScaler()
            final_X_train[feature_list] = scaler_features.fit_transform(
                final_X_train[feature_list]
            )
        else:
            scaler_features = None
        # 模型训练
        final_model = lgb.LGBMRegressor(**lgbm_params)
        final_model.fit(final_X_train, final_Y_train)

        return final_model, scaler_features, test_scores

    def forecast(
        self,
        lgb_model_workday,
        df_train_workday,
        # lgb_model_offday,
        # df_train_offday,
        scaler_features_workday,
        # scaler_features_offday,
    ):
        # 未来数据处理
        (df_future_workday, future_feature_list) = (
            self.process_future_data()
        )
        # 数据分割：工作日预测特征
        df_future_workday = df_future_workday.iloc[-self.horizon:, ]
        X_future_workday = df_future_workday.loc[:, future_feature_list]
        logger.info(
            f"{self.log_prefix} X_future_workday.head(): \n {X_future_workday.head()} \nX_future_workday length: {len(X_future_workday)} \nX_future_workday.columns: {X_future_workday.columns}"
        )
        # 数据分割：非工作日特征
        # X_future_offday = df_future_offday[future_feature_list]

        # 模型预测：multi-step recursive forecast
        if len(X_future_workday) > 0:
            if self.lags == []:
                Y_future_workday = lgb_model_workday.predict(X_future_workday)
            else:
                Y_future_workday = self._recursive_forecast(
                    model=lgb_model_workday,
                    history=df_train_workday,
                    future=X_future_workday,
                    lags=self.lags,
                    steps=min(self.horizon, len(X_future_workday)),
                    scaler_features=scaler_features_workday,
                )
            df_future_workday[self.target] = Y_future_workday
        logger.info(
            f"{self.log_prefix} df_future_workday: \n{df_future_workday.head()} \ndf_future_workday length after forecast: {len(df_future_workday)}"
        )

        '''
        # 模型预测：multi-step recursive forecast
        if len(X_future_offday) > 0:
            Y_future_offday = self._recursive_forecast(
                model=lgb_model_offday,
                history=df_train_offday,
                future=X_future_offday,
                lags=self.lags,
                steps=min(self.horizon, len(X_future_offday)),
                scaler_features=scaler_features_offday,
            )
            df_future_offday[self.target] = Y_future_offday
        logger.info(
            f"{self.log_prefix} df_future_offday is \n {df_future_offday.head(10)} \ndf_future_offday length: {len(df_future_offday)}"
        )
        '''
        # 输出结果处理
        # df_future = pd.merge(df_future, df_future_workday, how="outer")
        # df_future = pd.merge(df_future, df_future_offday, how="outer")
        df_future_workday.dropna(inplace=True, ignore_index=True)
        logger.info(
            f"{self.log_prefix} df_future.head(): \n{df_future_workday.head()}, \ndf_future length after dropna: {len(df_future_workday)}"
        )

        return df_future_workday

    def process_output(self, df_future):
        for i in range(len(df_future)):
            df_future.loc[i, "id"] = (
                f"{self.node_id}_{self.out_system_id}_{df_future.loc[i, 'timeStamp'].strftime('%Y%m%d%H%M%S')}"
            )
            df_future.loc[i, "node_id"] = self.node_id
            # 区分 in_system_id 和 out_system_id
            df_future.loc[i, "system_id"] = self.out_system_id
            df_future.loc[i, "predict_value"] = str(df_future.loc[i, self.target])
            # df_future.loc[i, "predict_adjustable_amount"] = str(
            #     df_future.loc[i, self.target] * random.uniform(0.05, 0.1)
            # )
            df_future.loc[i, "count_data_time"] = df_future.loc[
                i, "timeStamp"
            ].strftime("%Y-%m-%d %H:%M:%S.%f")[
                :-3
            ]  # 保留毫秒并精确到前3位

        df_future = df_future[
            [
                "id",
                "node_id",
                "system_id",
                "predict_value",
                # "predict_adjustable_amount",
                "count_data_time",
            ]
        ]

        return df_future

    def run(self, input_data: Dict, model_cfgs: Dict):
        """
        实际负荷预测
        """
        logger.info(f"{80*'='}")
        logger.info(f"Model Config...")
        logger.info(f"{80*'='}")
        # 参数
        self.now_time = model_cfgs["time_range"]["now_time"]
        self.start_time = model_cfgs["time_range"]["start_time"]
        self.future_time = model_cfgs["time_range"]["future_time"]
        self.lgbm_params_workday = model_cfgs["lgbm_params_workday"]
        self.lgbm_params_offday = model_cfgs["lgbm_params_offday"]
        self.node_id = model_cfgs["nodes"]["node"]["node_id"]
        self.out_system_id = model_cfgs["nodes"]["node"]["out_system_id"]
        self.target = "load"
        self.freq = "5min"
        self.input_data = input_data
        self.split_length = 288
        self.horizon = model_cfgs["time_range"]["after_days"] * 288 + 1
        self.lags = [
            # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
            # 96, 97, 98, 99, 100, 101, 
            # 192, 193, 194, 195, 196, 197, 
            # 288, 289, 290, 291, 292, 293
        ]
        self.scale = False
        logger.info(f"{self.log_prefix} start_time: {self.start_time}")
        logger.info(f"{self.log_prefix} now_time: {self.now_time}")
        logger.info(f"{self.log_prefix} future_time: {self.future_time}")
        # 历史数据处理
        logger.info(f"{80*'='}")
        logger.info(f"Model Training...")
        logger.info(f"{80*'='}")
        (df_load_workday, training_feature_list) = (
            self.process_history_data()
        )
        # df_load_workday.to_csv(f"./logs/test_results_{self.model}/df_load.csv", index=False)
        # 工作日模型训练
        data_X_workday = df_load_workday[training_feature_list]
        data_Y_workday = df_load_workday[self.target]
        lgb_model_workday, scaler_features_workday, test_scores = self.train(
            data_X=data_X_workday,
            data_Y=data_Y_workday,
            lgbm_params=self.lgbm_params_workday,
        )
        '''
        # 非工作日模型训练
        data_X_offday = df_load_offday[training_feature_list]
        data_Y_offday = df_load_offday[self.target]
        lgb_model_offday, scaler_features_offday = self.train(
            data_X=data_X_offday,
            data_Y=data_Y_offday,
            lgbm_params=self.lgbm_params_offday,
        )
        '''
        # 模型预测
        logger.info(f"{80*'='}")
        logger.info(f"Model Forecast...")
        logger.info(f"{80*'='}")
        df_future = self.forecast(
            lgb_model_workday=lgb_model_workday,
            df_train_workday=df_load_workday,
            # lgb_model_offday=lgb_model_offday,
            # df_train_offday=df_load_offday,
            scaler_features_workday=scaler_features_workday,
            # scaler_features_offday=scaler_features_offday,
        )
        # 输出结果处理
        logger.info(f"{80*'='}")
        logger.info(f"Forecast result processing...")
        logger.info(f"{80*'='}")
        df_power_future = self.process_output(df_future)

        # 模型输出
        return {"df_future": df_power_future}, test_scores




# 测试代码 main 函数
def main():
    import datetime
    from pathlib import Path
    # ------------------------------
    # model configs
    # ------------------------------
    # pred_method = "multip-step-directly"          # 预测方法
    # scale = False                                 # 是否进行标准化
    # power_timestamp_feat = "time"                 # 功率数据时间戳特征名称
    # date_timestamp_feat = "time"                  # 日期数据时间戳特征名称
    # weather_timestamp_feat = "time"               # 天气数据时间戳特征名称
    # target = "value"                              # 预测目标变量名称
    # target_transform = False                      # 预测目标是否需要转换
    # target_transform_predict = None               # 预测目标转换是否需要预测
    # freq = "5min"                                 # 数据频率
    # n_per_day = 24 * 12                           # 每天样本数量
    # lags = []                                     # 特征滞后数列表
    # n_lags = len(lags)                            # 特征滞后数个数(1,2,...)
    history_days = 30                             # 历史数据天数
    predict_days = 1                              # 预测未来1天的功率
    # window_days = 15                              # 滑动窗口天数
    # horizon = predict_days * n_per_day            # 预测未来 1天(24小时)的功率/数据划分长度/预测数据长度
    # n_windows = history_days - (window_days - 1)  # 测试滑动窗口数量, >=1, 1: 单个窗口
    # window_len = window_days * n_per_day if n_windows > 1 else history_days * n_per_day   # 测试窗口数据长度(训练+测试)
    # date_type = None                                                                      # 日期类型
    test_scores_df = pd.DataFrame()
    for now in pd.date_range("2025-10-31 00:00:00", "2025-12-27 00:00:00", freq="1d"):
        print(now)
        # now = datetime.datetime(2025, 12, 31, 0, 0, 0)                                      # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)                # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)         # 时间序列历史数据开始时刻
        future_time = now_time + datetime.timedelta(days=predict_days)                        # 时间序列未来结束时刻
        logger.info(f"\nhistory data: {start_time} ~ {now_time} \npredict data: {now_time} ~ {future_time}")
        # 模型参数
        model_cfgs = {
            # "pred_method": pred_method,
            # "scale": scale,
            # "power_timestamp_feat": power_timestamp_feat,
            # "date_timestamp_feat": date_timestamp_feat,
            # "weather_timestamp_feat": weather_timestamp_feat,
            # "target": target,
            # "target_transform": target_transform,
            # "target_transform_predict": target_transform_predict,
            # "freq": freq,
            # "lags": lags,
            # "n_lags": n_lags,
            # "n_per_day": n_per_day,
            # "window_days": window_days,
            # "horizon": horizon,
            # "n_windows": n_windows,
            # "window_len": window_len,
            # "date_type": date_type,
            "nodes": {
                "node": {
                    "node_id": 1,
                    "out_system_id": 1,
                }
            },
            "time_range": {
                "start_time": start_time,
                "now_time": now_time,
                "future_time": future_time,
                "before_days": -history_days,
                "after_days": predict_days,
            },
            "lgbm_params_workday": {
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
            },
            "lgbm_params_offday": {
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
            },
        }
        # ------------------------------
        # get data
        # ------------------------------
        data_dir = Path("./model/model_packages/DemandLoad_lingang/dataset/electricity/2026-01-01/lingang/demand_load/lingang_B/")
        df_power = pd.read_csv(data_dir.joinpath(f"AIDC_B_dataset.csv"))
        df_date = pd.read_csv(data_dir.joinpath("df_date.csv"))
        df_weather = pd.read_csv(data_dir.joinpath("df_weather.csv"))
        df_date_future = pd.read_csv(data_dir.joinpath("df_date_future.csv"))
        df_weather_future = pd.read_csv(data_dir.joinpath("df_weather_future.csv"))
        df_date_all = pd.concat([df_date.iloc[:-1, ], df_date_future], axis=0)
        df_weather_all = pd.concat([df_weather.iloc[:-1,], df_weather_future], axis=0)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_date_all)
            # print(df_weather_all)
        input_data = {
            "df_power": df_power,
            "df_date": df_date_all,
            "df_weather": df_weather_all,
            "df_date_future": df_date_all,
            "df_weather_future": df_weather_all,
        }
        # ------------------------------
        # 模型测试
        # ------------------------------
        model_ins = ModelMainClass(
            project="test",
            model="test",
            node="test",
            args={},
        )
        result, test_scores = model_ins.run(input_data, model_cfgs)
        test_scores_df_temp = pd.DataFrame(test_scores, index=[now.date()])
        test_scores_df = pd.concat([test_scores_df, test_scores_df_temp], axis=0)
    
    test_scores_df.to_csv("test_scores_df.csv", encoding="utf-8", index=True)

if __name__ == "__main__":
    main()

