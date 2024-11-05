# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ExpandingFeatures.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042415
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ExpandingFeatures:
    
    def __init__(self, 
                 data, 
                 datetime_format: str = '%Y-%m-%d %H:%M:%S',
                 window_length: int = 7) -> None:
        self.data = data
        self.datetime_format = datetime_format
        self.window_length = window_length

    def features(self, raw_feature, new_feature):
        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"], format = self.datetime_format)
        # TODO
        self.data[new_feature] = self.data[raw_feature].expanding(2).mean()
        # 重命名
        data_columns = ["Datetime", raw_feature, new_feature]
        
        self.data = self.data[data_columns]




# 测试代码 main 函数
def main():
    # 数据读取
    series = pd.read_csv(
        "D:/projects/timeseries_forecasting/tsproj/dataset/daily-minimum-temperatures-in-me.csv",
        header = 0,
        index_col = 0,
        # parse_dates = [0],
        # date_parser = lambda dates: pd.to_datetime("190" + dates, format = "%Y-%m"),
    )
    print(series.head())
    temps = pd.DataFrame(series.values)
    print(temps.head())
    # 使用 expanding 方法对先前所有值进行统计
    window = temps.expanding()
    df = pd.concat([
        window.min(), 
        window.mean(), 
        window.max(), 
        temps.shift(0)
    ], axis = 1)
    df.columns = ["min", "mean", "max", "t+1"]
    print(df)

if __name__ == "__main__":
    main()
