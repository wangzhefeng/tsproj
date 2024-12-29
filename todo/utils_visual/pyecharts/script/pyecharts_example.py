# -*- coding: utf-8 -*-

__author__ = 'wangzhefeng'



##################### download and install the library ######################
# $pip install pyecharts

# $ git clone --recursive https://github.com/chenjiandongx/pyecharts.git
# $ cd pyecharts
# $ python setup.py install
#############################################################################
# pyecharts 预览模式
# 1 chart.render(r"/home/wangzhefeng/project/DataVisualiztion/pyecharts/")
# 2 使用 pyecharts-snapshot 插件
#   $sudo su
# 	$npm install -g phantomjs-prebuilt
#   $pip install pyecharts-snapshot
#   (1) snapshot output.html [png|jpeg|gif|pdf] delay_in_seconds
#   (2) from pyecharts_snapshot.main import make_a_snapshot
#       make_a_snapshot("render.html", "snapshot.png/pdf/gif", delay = 1)
# 3 $jupyter notebook
#
#############################################################################

import pyecharts
from pyecharts import Bar, Grid
from pyecharts_snapshot.main import make_a_snapshot
print(pyecharts.__version__)

#-------------------------------------------------------------------------------------
# Basic Usage

attr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
		"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 
	  135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 
	  175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
bar = Bar("Bar chart", 
		  "precipitation and evaporation one year")
bar.add("precipitation", 
		attr, v1, 
		mark_line = ["average"], 
		mark_point = ["max", "min"])
bar.add("evaporation", 
		attr, v2, 
		mark_line = ["average"], 
		mark_point = ["max", "min"])
bar.render(r"/home/wangzhefeng/project/DataVisualiztion/pyecharts/bar_chart1.html")

#-------------------------------------------------------------------------------------
# Working with pandas and numpy

import pandas as pd
import numpy as np

title = "bar chart"
index = pd.date_range("3/8/2017", periods = 6, freq = "M")

df1 = pd.DataFrame(np.random.randn(6), index = index)
df2 = pd.DataFrame(np.random.randn(6), index = index)

dtvalue1 = [i[0] for i in df1.values]
dtvalue2 = [i[0] for i in df2.values]
_index = [i for i in df1.index.format()]

bar = Bar(title, "Profit and loss situation")
bar.add("profit", _index, dtvalue1)
bar.add("loss", _index, dtvalue2)
bar.render(r"/home/wangzhefeng/project/DataVisualiztion/pyecharts/bar_chart2.html")

#-------------------------------------------------------------------------------------
# My first Echarts

bar = Bar("我的第一个图表", "这里是副标题")
bar.add("服装", 
		["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90],
		is_more_utils = True)
bar.show_config()
bar.render(r"/home/wangzhefeng/project/DataVisualiztion/pyecharts/my_first_chart.html")
