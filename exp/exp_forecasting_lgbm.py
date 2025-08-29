# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-29
# * Version     : 1.0.082910
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# plt.rcParams['font.sans-serif']=['SimHei', 'Arial Unicode MS']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# --- 1. 数据模拟 ---
# ------------------------------
# 在实际应用中，您应该从CSV或其他文件中加载您的真实数据
# 例如: df = pd.read_csv('your_data.csv')
# 这里我们创建一个与您描述结构相同的模拟数据集来进行演示
logger.info("步骤1：正在创建模拟数据...")
logger.info(f"{'-' * 40}")
# 创建一个时间序列，假设我们有10天的历史数据
rng = pd.date_range('2025-07-01 00:00:00', periods=10 * 288, freq='5T')
df = pd.DataFrame({'data_time': rng})

# 创建特征
# 模拟一个有每日和每周周期的电力负荷
base_load = 2000
daily_cycle = 500 * np.sin(np.arange(len(df)) * 2 * np.pi / 288)
weekly_cycle = 200 * np.sin(np.arange(len(df)) * 2 * np.pi / (288 * 7))
noise = np.random.normal(0, 50, len(df))
df['real_power'] = base_load + daily_cycle + weekly_cycle + noise

# 模拟天气数据
df['wind_speed'] = np.random.uniform(0, 15, len(df))
df['wind_angle'] = np.random.uniform(0, 360, len(df))
df['temp'] = 25 + 5 * np.sin(np.arange(len(df)) * 2 * np.pi / 288) + np.random.uniform(-2, 2, len(df))
df['prec'] = np.random.choice([0, 0.1, 0.5, 1], len(df), p=[0.9, 0.05, 0.03, 0.02])
df['pressure'] = 1010 + np.random.uniform(-5, 5, len(df))
df['clouds'] = np.random.randint(0, 101, len(df))
df['feels_like'] = df['temp'] + np.random.uniform(-1, 1, len(df))
df['rh'] = np.random.randint(40, 90, len(df))
df['weather_code'] = np.random.choice([40, 41, 42], len(df), p=[0.7, 0.2, 0.1])

logger.info("模拟数据创建完成。")
logger.info("数据预览：")
logger.info(df)
logger.info(df.columns)


# ------------------------------
# --- 2. 特征工程 ---
# ------------------------------
logger.info("\n步骤2：正在进行特征工程...")
logger.info(f"{'-' * 40}")

def create_features(data_df):
    """
    从数据框中创建时间序列特征
    """
    data_df = data_df.copy()
    data_df['data_time'] = pd.to_datetime(data_df['data_time'])

    # 时间基础特征
    data_df['hour'] = data_df['data_time'].dt.hour
    data_df['minute'] = data_df['data_time'].dt.minute
    data_df['day_of_week'] = data_df['data_time'].dt.dayofweek # 0=周一, 6=周日
    data_df['day_of_year'] = data_df['data_time'].dt.dayofyear
    data_df['month'] = data_df['data_time'].dt.month
    data_df['quarter'] = data_df['data_time'].dt.quarter
    
    # 周期性特征 (将时间转换为可循环的 sin/cos 形式)
    data_df['minute_in_day'] = data_df['hour'] * 12 + data_df['minute'] / 5
    data_df['minute_in_day_sin'] = np.sin(data_df['minute_in_day'] * (2 * np.pi / 288))
    data_df['minute_in_day_cos'] = np.cos(data_df['minute_in_day'] * (2 * np.pi / 288))
    
    # 滞后特征 (Lag Features) - 过去时间的负荷值
    # 我们将使用1天前和2天前的同一时刻的负荷作为特征
    # 注意：在真实预测时，这些滞后值需要被逐步填充
    lag_days = [1, 2, 7] # 1天前, 2天前，7天前（上周）
    for lag in lag_days:
        data_df[f'lag_{lag*288}'] = data_df['real_power'].shift(lag * 288)

    return data_df

df_featured = create_features(df)
# 移除因创建滞后特征而产生的空值行
df_featured = df_featured.dropna()
logger.info("特征工程完成。")
logger.info("添加特征后的数据预览：")
logger.info(df_featured.head())

# ------------------------------
# --- 3. 模型训练 ---
# ------------------------------
logger.info("\n步骤3：正在训练LightGBM模型...")
logger.info(f"{'-' * 40}")

# 定义特征列和目标列
TARGET = 'real_power'
# 'weather_code' 作为分类特征处理
CATEGORICAL_FEATURES = ['weather_code', 'hour', 'day_of_week', 'month', 'quarter']
FEATURES = [col for col in df_featured.columns if col not in ['data_time', TARGET, 'date', 'DATA_TIME', 'DAY']]

# 划分数据（这里我们使用所有可用历史数据进行训练）
X_train = df_featured[FEATURES]
y_train = df_featured[TARGET]

# 创建并训练LightGBM模型
params = {
    'objective': 'regression_l1', # L1损失，对异常值不敏感
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**params)

# 训练模型，并告知哪些是分类特征
model.fit(X_train, y_train, 
          categorical_feature=CATEGORICAL_FEATURES,
          eval_set=[(X_train, y_train)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(100, verbose=False)])

logger.info("模型训练完成。")


# ------------------------------
# --- 4. 递归预测未来288个点 ---
# ------------------------------
logger.info("\n步骤4：正在进行未来288个点的递归预测...")
logger.info(f"{'-' * 40}")

# 获取用于预测的最后一段历史数据
history_df = df.copy()

# 创建未来288个点的时间戳
future_dates = pd.date_range(history_df['data_time'].iloc[-1] + pd.Timedelta(minutes=5), periods=288, freq='5T')
# 假设未来的天气数据是已知的（天气预报）,在此我们用历史最后一天的数据来模拟未来的天气预报
future_weather_forecast = history_df.iloc[-288:].copy()
future_weather_forecast['data_time'] = future_dates
future_weather_forecast = future_weather_forecast.drop(columns=['real_power'])
logger.info(future_weather_forecast)

# 存储预测结果
predictions = []
for i in range(288):
    # 1. 准备当前要预测的时间点的数据框
    current_step_df = future_weather_forecast.iloc[i:i+1].copy()
    logger.info(f"debug::current_step_df: \n{current_step_df}")
    
    # 2. 将历史数据和当前步的数据合并，以便创建特征
    combined_df = pd.concat([history_df, current_step_df], ignore_index=True)
    logger.info(f"debug::combined_df: \n{combined_df}")
    
    # 3. 为合并后的数据创建特征
    combined_featured = create_features(combined_df)
    logger.info(f"debug::combined_featured: \n{combined_featured}")
    
    # 4. 提取出当前预测步所需要的特征（最后一行）
    current_features = combined_featured[FEATURES].iloc[-1:]
    logger.info(f"debug::current_features: \n{current_features}")
    
    # 5. 进行预测
    prediction = model.predict(current_features)[0]
    predictions.append(prediction)
    
    # 6. 【关键】将预测值更新回history_df，以便为下一步预测提供滞后特征
    # 创建一个包含预测值的新行
    new_row = current_step_df.copy()
    new_row['real_power'] = prediction
    # 将新行添加到历史数据中，进行下一次循环
    history_df = pd.concat([history_df, new_row], ignore_index=True)
logger.info("预测完成。")

# 创建包含预测结果的DataFrame
future_predictions_df = pd.DataFrame({
    'data_time': future_dates,
    'predicted_power': predictions
})
logger.info("\n未来一天288点负荷预测结果：")
logger.info(future_predictions_df)


# ------------------------------
# --- 5. 结果可视化 ---
# ------------------------------
logger.info("\n步骤5：正在生成结果可视化图表...")
logger.info(f"{'-' * 40}")

plt.figure(figsize=(18, 8))
# 绘制历史最后两天的真实负荷
historical_to_plot = df.iloc[-2 * 288:]
plt.plot(historical_to_plot['data_time'], historical_to_plot['real_power'], label='历史实际负荷', color='blue')
# 绘制预测负荷
plt.plot(future_predictions_df['data_time'], future_predictions_df['predicted_power'], label='预测负荷', color='red', linestyle='--')
plt.title('电力负荷预测：历史数据与未来24小时预测', fontsize=16)
plt.xlabel('时间', fontsize=12)
plt.ylabel('负荷 (real_power)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

logger.info("\n所有步骤执行完毕。")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
