# -*- coding: utf-8 -*-

# ***************************************************
# * File        : periodic_factor_forecast.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-10-25
# * Version     : 0.1.102521
# * Description : 时间序列预测周期因子法
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


df = pd.DataFrame({
    "ts": pd.date_range("2022-10-04", periods = 21, freq = "D"),
    "value": [
        20, 10, 70, 50, 250, 200, 100, 
        26, 18, 66, 50, 180, 140, 80,
        15, 8, 67, 60, 270, 160, 120,
    ],
})
# fig, ax = plt.subplots()
# plt.plot(df["ts"], df["value"])
# plt.title("Value")
# plt.show()

df["weekday"] = df["ts"].dt.weekday
df["day"] = df["ts"].dt.day
df["week"] = df["ts"].dt.week
df["month"] = df["ts"].dt.month
logger.info(df)



def generate_base(df: pd.DataFrame, month_index: int) -> pd.DataFrame:
    # 选中固定时间段的数据集
    total_balance = df.copy()
    total_balance = total_balance[['date', 'total_purchase_amt', 'total_redeem_amt']]
    total_balance = total_balance[(total_balance['date'] >= datetime.date(2014,3,1)) & (total_balance['date'] < datetime.date(2014, month_index, 1))]

    # 加入时间戳
    total_balance['weekday'] = total_balance['date'].dt.weekday
    total_balance['day'] = total_balance['date'].dt.day
    total_balance['week'] = total_balance['date'].dt.week
    total_balance['month'] = total_balance['date'].dt.month
    
    # 统计翌日因子
    mean_of_each_weekday = total_balance[['weekday']+['total_purchase_amt','total_redeem_amt']].groupby('weekday',as_index=False).mean()
    for name in ['total_purchase_amt','total_redeem_amt']:
        mean_of_each_weekday = mean_of_each_weekday.rename(columns={name: name+'_weekdaymean'})
    mean_of_each_weekday['total_purchase_amt_weekdaymean'] /= np.mean(total_balance['total_purchase_amt'])
    mean_of_each_weekday['total_redeem_amt_weekdaymean'] /= np.mean(total_balance['total_redeem_amt'])

    # 合并统计结果到原数据集
    total_balance = pd.merge(total_balance, mean_of_each_weekday, on='weekday', how='left')

    # 分别统计翌日在(1~31)号出现的频次
    weekday_count = total_balance[['day','weekday','date']].groupby(['day','weekday'],as_index=False).count()
    weekday_count = pd.merge(weekday_count, mean_of_each_weekday, on='weekday')

    # 依据频次对翌日因子进行加权，获得日期因子
    weekday_count['total_purchase_amt_weekdaymean'] *= weekday_count['date']   / len(np.unique(total_balance['month']))
    weekday_count['total_redeem_amt_weekdaymean'] *= weekday_count['date']  / len(np.unique(total_balance['month']))
    day_rate = weekday_count.drop(['weekday','date'],axis = 1).groupby('day',as_index=False).sum()

    # 将训练集中所有日期的均值剔除日期残差得到base
    day_mean = total_balance[['day'] + ['total_purchase_amt','total_redeem_amt']].groupby('day',as_index=False).mean()
    day_pre = pd.merge(day_mean, day_rate, on='day', how='left')
    day_pre['total_purchase_amt'] /= day_pre['total_purchase_amt_weekdaymean']
    day_pre['total_redeem_amt'] /= day_pre['total_redeem_amt_weekdaymean']

    # 生成测试集数据
    for index, row in day_pre.iterrows():
        if month_index in (2,4,6,9) and row['day'] == 31:
            break
        day_pre.loc[index, 'date'] = datetime.datetime(2014, month_index, int(row['day']))

    # 基于base与翌日因子获得最后的预测结果
    day_pre['weekday'] = day_pre.date.dt.weekday
    day_pre = day_pre[['date','weekday']+['total_purchase_amt','total_redeem_amt']]
    day_pre = pd.merge(day_pre, mean_of_each_weekday,on='weekday')
    day_pre['total_purchase_amt'] *= day_pre['total_purchase_amt_weekdaymean']
    day_pre['total_redeem_amt'] *= day_pre['total_redeem_amt_weekdaymean']

    day_pre = day_pre.sort_values('date')[['date']+['total_purchase_amt','total_redeem_amt']]
    return day_pre




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

