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
from pyecharts import Bar, Line, Pie, Grid, Overlap
from pyecharts_snapshot.main import make_a_snapshot
print(pyecharts.__version__)


