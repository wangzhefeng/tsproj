import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL


def extract_periodic_features(df, target_column, time_column=None, max_lags=2000, seasonal_period=960):
    """
    提取时间序列的周期性特征。

    参数:
    - df: pandas.DataFrame，包含时间序列数据。
    - target_column: str，目标列名（需要分析的时间序列列）。
    - time_column: str，时间列名（可选，如果提供则用于索引）。
    - max_lags: int，自相关分析的最大滞后值。
    - seasonal_period: int，STL分解的季节性周期。

    返回:
    - features_df: pandas.DataFrame，包含提取的周期性特征。
    """
    # 如果提供了时间列，则设置为索引
    if time_column:
        df = df.set_index(time_column)

    # 提取目标时间序列
    data = df[target_column].values

    # 初始化结果 DataFrame
    features_df = pd.DataFrame(index=df.index)

    # 1. 傅里叶变换提取周期性特征
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, d=1)  # 假设时间间隔为 1
    dominant_freq = xf[np.argmax(np.abs(yf[:n // 2]))]
    features_df['dominant_frequency'] = dominant_freq

    # 2. 自相关分析提取周期性特征
    autocorr = acf(data, nlags=max_lags)
    peaks = np.where(autocorr > 0.5)[0]
    if len(peaks) > 1:
        period = peaks[1] - peaks[0]
    else:
        period = np.nan
    features_df['autocorrelation_period'] = period

    # 3. STL分解提取季节性成分
    try:
        stl = STL(data, seasonal=seasonal_period, period=seasonal_period)  # 明确指定 period 参数
        result = stl.fit()
        features_df['seasonal_component'] = result.seasonal
    except ValueError as e:
        print(f"STL decomposition failed: {e}")
        features_df['seasonal_component'] = np.nan  # 如果失败，填充为 NaN

    return features_df




# 示例使用
if __name__ == "__main__":
    # 构造示例数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = 0.1 * np.arange(100) + np.sin(2 * np.pi * np.arange(100) / 12) + 0.2 * np.random.randn(100)
    df = pd.DataFrame({'date': dates, 'value': data})
    print(df.head())

    # 提取周期性特征
    features_df = extract_periodic_features(df, target_column='value', time_column='date')

    # 打印结果
    print(features_df.head())
