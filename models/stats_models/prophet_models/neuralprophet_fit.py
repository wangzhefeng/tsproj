# -*- coding: utf-8 -*-

# ***************************************************
# * File        : neuralprophet_fit.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-10
# * Version     : 1.0.091022
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
    
import pandas as pd
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

set_random_seed(0)
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
df = pd.read_csv(data_location + "wp_log_peyton_manning.csv")
print(df.head())
print(df.shape)

# model
m = NeuralProphet()

# train and test data
df_train, df_test = m.split_df(df, valid_p = 0.2)

# model train
metrics = m.fit(df_train, validation_df = df_test)

# model validation
# m = NeuralProphet()
# train_metrics = m.fit(df_trian)
# test_metrics = m.fit(df_test)

# model validation metrics
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(metrics)

# model predict
predicted = m.predict(df)
forecast = m.predict(df)
print(predicted)
print(forecast)

# plotting
forecast_plot = m.plot(forecast)
fig_comp = m.plot_components((forecast))
fig_param = m.plot_parameters()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
