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
# 3 
#############################################################################

import pyecharts
from pyecharts import Line, Pie, Bar, Grid
from pyecharts_snapshot.main import make_a_snapshot
print(pyecharts.__version__)

# Line
line = Line("折线图示例", width = 1200)
attr_line = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
data1_line = [11, 11, 15, 13, 12, 13, 10]
data2_line = [1, -2, 2, 5, 3, 2, 0]
line.add("最高气温", 
		 attr_line, data1_line, 
		 mark_point = ["max", "min"], 
		 mark_line = ["average"])
line.add("最低气温", 
		 attr_line, data2_line,
		 mark_point = ["max", "min"], 
		 mark_line = ["avergae"], 
		 legend_pos = "20%")

# Pie
attr_pie = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
data_pie = [11, 12, 13, 10, 10, 10]
pie = Pie("饼图示例", title_pos = "45%")
pie.add("", 
		attr_pie, data_pie, 
		radius = [30, 55],
		legend_pos = "65%",
		legend_orient = "horizonial")

# Grid
grid = Grid()
grid.add(line, grid_right = "65%")
grid.add(pie, grid_left = "60%")
grid.render("/home/wangzhefeng/project/DataVisualiztion/pyecharts/grid_test.html")
make_a_snapshot("grid_test.html", "grid_test.pdf")
