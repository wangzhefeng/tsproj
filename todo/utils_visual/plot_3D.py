# -*- coding: utf-8 -*-

# ***************************************************
# * File        : plot_3D.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-12-29
# * Version     : 0.1.122921
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def scatter_3d_plot(df, x, y, z):
    fig = go.Figure(
        data = [
            go.Scatter3d(
                x = df[x],
                y = df[x],
                z = df[x],
                mode = 'markers',
                marker = dict(
                    size = 3,
                    # color=z,                # set color to an array/list of desired values
                    # colorscale='Viridis',   # choose a colorscale
                    opacity = 0.8
                )
            )
        ]
    )
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def scatter_3d_plot(df, x, y, z):
    fig = plt.figure(figsize = (20, 8))
    # Plotly Express
    fig = px.scatter_3d(
        df, 
        x = x, 
        y = y, 
        z = z
    )
    # tight layout
    fig.update_layout(margin = dict(l = 3, r = 3, b = 3, t = 3))
    fig.show()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
