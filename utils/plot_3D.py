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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import plotly.express as px
import plotly.graph_objects as go

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


"""
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Iris samples filtered by petal width'),
    dcc.Graph(id="3d-scatter-plot-x-graph"),
    html.P("Petal Width:"),
    dcc.RangeSlider(
        id='3d-scatter-plot-x-range-slider',
        min=0, 
        max=2.5, 
        step=0.1,
        marks={0: '0', 2.5: '2.5'},
        value=[0.5, 2]
    ),
])

@app.callback(
    Output("3d-scatter-plot-x-graph", "figure"),
    Input("3d-scatter-plot-x-range-slider", "value"))
def update_bar_chart(slider_range):
    df = px.data.iris() # replace with your own data source
    low, high = slider_range
    mask = (df.petal_width > low) & (df.petal_width < high)

    fig = px.scatter_3d(
        df[mask],
        x='sepal_length', 
        y='sepal_width', 
        z='petal_width',
        color="species", 
        hover_data=['petal_width']
    )
    
    return fig

app.run(debug = True)
"""


def scatter_3d_plot_go(df, x, y, z):
    fig = go.Figure(
        data = [
            go.Scatter3d(
                x = df[x],
                y = df[y],
                z = df[z],
                mode = 'markers',
                marker = dict(
                    size = 1,
                    # color=z,  # set color to an array/list of desired values
                    # colorscale='Viridis',  # choose a colorscale
                    opacity = 0.8
                ),
            )
        ],
    )
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def scatter_3d_plot_px(df, x, y, z, grad_col = None, cate_col = None):
    # Plotly Express
    fig = px.scatter_3d(
        df,
        x = x,
        y = y,
        z = z,
        color=grad_col,
        symbol=cate_col,
        size=[1] * len(df),
        size_max=7,
        opacity=0.8,
        width=1000,
        height=1000,
    )
    # tight layout
    fig.update_layout(margin = dict(l=0, r=0, b=0, t=0))
    fig.show()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
