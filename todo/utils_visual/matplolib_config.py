import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.abspath(".."))


# ------------
# jupyter lab
# ------------
# %matplotlib inline

# ------------
# 设置图片主题
# ------------
matplotlib.style.use("ggplot")
plt.style.use("fivethirtyeight")

# ------------
# 解决 Mac 中文显示问题
# ------------
from matplotlib.font_manager import FontProperties

# ------------
# 设置figure_size尺寸
# ------------
plt.rcParams['figure.figsize'] = (4.0, 4.0)

# ------------
# 设置 interpolation style
# ------------
plt.rcParams['image.interpolation'] = 'nearest'

# ------------
# 设置颜色风格
# ------------
plt.rcParams['image.cmap'] = 'gray'

# ------------
# 设置图片像素
# ------------
plt.rcParams['savefig.dpi'] = 100

# ------------
# 设置图片分辨率
# ------------
plt.rcParams['figure.dpi'] = 100

# ------------
# 设置图片正常显示中文
# ------------
plt.rcParams['font.family'] = ['Arial Unicode MS']
