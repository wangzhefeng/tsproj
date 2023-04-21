# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-05-05
# * Version     : 0.1.050523
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from autots import AutoTS, load_daily

# data
long = False
df = load_daily(long = long)
print(df.head())


# model
model = AutoTS(
    forecast_length = 21,
    frequency = "infer",
    prediction_interval = 0.9,
    ensemble = None,
    model_list = "fast",  # "superfast", "defalut", "fast_parallel"
    transformer_list = "fast",  # "superfast"
    drop_most_recent = 1,
    max_generations = 4,
    num_validations = 2,
    validation_method = "backwards",
)
model = model.fit(
    df,
    date_col = "datetime" if long else None,
    value_col = "value" if long else None,
    id_col = "series_id" if long else None,
)
prediction = model.predict()


# plot
prediction.plot(
    model.df_wide_numeric,
    series = model.df_wide_numeric.columns[0],
    start_date = "2019-01-01",
)

# best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast

# upper and lower forecast
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_result = model.results()

# add aggregated from cross validation
validation_results = model.results("validation")














# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

