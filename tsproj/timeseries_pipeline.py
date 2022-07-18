
# -*- coding: utf-8 -*-


"""
1. 什么是时间序列?
2. 如何在Python中导入时间序列?
3. 什么是面板数据?
4. 时间序列可视化
5. 时间序列的模式
6. 时间序列的加法和乘法
7. 如何将时间序列分解?
8. 平稳和非平稳时间序列
9. 如何获取平稳的时间序列?
10. 如何检验平稳性?
11. 白噪音和平稳序列的差异是什么?
12. 如何去除时间序列的线性分量?
13. 如何消除时间序列的季节性?
14. 如何检验时间序列的季节性?
15. 如何处理时间序列中的缺失值?
16. 什么是自回归和偏自回归函数?
17. 如何计算偏自回归函数?
18. 滞后图
19. 如何估计时间序列的预测能力?
20. 为什么以及怎样使时间序列平滑?
21. 如何使用Granger因果检验来获知时间序列是否对预测另一个序列帮助?
22. 下一步是什么?
"""


from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 120,
})


# # 1.数据


# ## 1.1 时间序列数据


df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/a10.csv", parse_dates = ["date"])
df.head()


# ## 1.2 时间序列数据


ser = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/a10.csv", parse_dates = ["date"], index_col = "date")
ser.head()


# ## 2.3 面板数据


pan_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv")
pan_df = df.loc[df.market == "MUMBAI", :]
pan_df.head()





