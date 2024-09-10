# -*- coding: utf-8 -*-

# ***************************************************
# * File        : midimax_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-07
# * Version     : 0.1.090722
# * Description : description
# * Link        : https://github.com/edwinsutrisno/midimax_compression
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, save

from midimax import compress_series

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# Create a time-series of sine wave
n = 1000  # points
timesteps = pd.to_timedelta(np.arange(n), unit = 's')
timestamps = pd.to_datetime("2022-04-18 08:00:00") + timesteps

sine_waves = np.sin(2 * np.pi * 0.02 * np.arange(n))
noise = np.random.normal(0, 0.1, n)
signal = sine_waves + noise
ts_data = pd.Series(signal, index = timestamps).astype('float32')
print(f"ts_data:\n{ts_data}")

# Run compression
timer_start = time.time()
ts_data_compressed = compress_series(ts_data, 2)
timer_sec = round(time.time() - timer_start, 2)
print(f"\nts_data_compressed:\n{ts_data_compressed}")
print(f"Compression took {timer_sec} seconds.")


def format_fig_axis(fig):
    """
    Formatting the date stamps on the plot axis
    """
    fig.xaxis.formatter = DatetimeTickFormatter(
        days = ["%m/%d %H:%M:%S"],
        months = ["%m/%d %H:%M:%S"],
        hours = ["%m/%d %H:%M:%S"],
        minutes = ["%m/%d %H:%M:%S"]
    )
    fig.xaxis.axis_label = 'Timestamp'
    fig.yaxis.axis_label = 'Series Value'

    return fig


# Plot before
fig1 = figure(sizing_mode = 'stretch_both', tools = 'box_zoom,pan,reset')
line_before = fig1.line(
    x = ts_data.index, 
    y = ts_data.values, 
    line_width = 2
)
fig1 = format_fig_axis(fig1)
output_file('ts_visual/results/demo_output_before_compression.html')
save(fig1)


# Plot after
fig2 = figure(sizing_mode = 'stretch_both', tools = 'box_zoom,pan,reset')
line_after = fig2.line(
    x = ts_data_compressed.index, 
    y = ts_data_compressed.values, 
    line_color = 'green'
)
fig2 = format_fig_axis(fig2)
output_file('ts_visual/results/demo_output_after_compression.html')
save(fig2)


# Plot before and after together
fig3 = figure(sizing_mode = 'stretch_both', tools = 'box_zoom,pan,reset')
fig3.line(
    x = ts_data.index, 
    y = ts_data.values, 
    line_width = 2
)
fig3.line(
    x = ts_data_compressed.index, 
    y = ts_data_compressed.values, 
    line_color = 'green', 
    line_dash = 'dashed'
)
fig3.scatter(
    x = ts_data_compressed.index, 
    y = ts_data_compressed.values, 
    marker = 'circle', 
    size = 8, 
    color = 'green'
)
fig3 = format_fig_axis(fig3)
output_file('ts_visual/results/demo_output_before_and_after_compression.html')
save(fig3)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
