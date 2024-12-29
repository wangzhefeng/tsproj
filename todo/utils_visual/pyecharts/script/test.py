# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from pyecharts.charts import Bar

bar = Bar()
bar.add_xaxis([
	"衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"
])
bar.add_yaxis("商家", [5, 20, 36, 10, 75, 90])
bar.render("../html/myecharts.html")
