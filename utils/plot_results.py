# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-22
# * Version     : 0.1.052220
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
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_results(preds, trues, title: str = "result"):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(trues, label = 'True Data')
    plt.plot(preds, label = 'Prediction')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(f'images/{title}_results.png')


def plot_results_multiple(preds, trues, preds_len: int, title: str = "results_multiple"):
    """
    绘图展示结果
    """
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(trues, label = 'True Data')
    for i, data in enumerate(preds):
        padding = [None for p in range(i * preds_len)]
        plt.plot(padding + data, label = 'Prediction')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(f'images/{title}_results_multiple.png')


def predict_result_visual(preds, trues, path='./pic/test.pdf'):
    """
    Results visualization
    """
    # 设置绘图风格
    # plt.style.use('ggplot')
    # 画布
    fig = plt.figure(figsize = (25, 5))
    # 创建折线图
    plt.plot(trues, label='Trues', lw=1, color="blue")
    plt.plot(preds, label='Preds', lw=1, color="read", linestyle="--")
    # 增强视觉效果
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title('Trues and Preds Timeseries Plot')
    # plt.ylim(5, 20)
    # plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.show();


def series_plot(df, col):
    plt.figure(figsize=(15, 5))
    plt.plot(df["ds"], df[col])
    plt.xlabel("Date [H]")
    plt.ylabel("有功功率")
    plt.title("有功功率")
    # plt.legend()
    plt.tight_layout()
    plt.grid()


def plot_cv_predictions(cv_plot_df, device_data, data_type = "split"):
    # history ture data
    history_data = device_data["history_data"].dropna()
    history_data.set_index("ds", inplace=True)
    history_data = history_data.loc[history_data.index >= min(cv_plot_df.index), ]
    # future predict data
    predict_data = device_data["predict_data"].dropna()
    predict_data.set_index("ds", inplace=True)
    # plot
    for col_true, col_pred in zip(
        history_data.columns, 
        [col for col in cv_plot_df.columns if col not in ["train_start", "cutoff", "test_end"]]
    ):
        # 画布
        plt.figure(figsize = (15, 5))
        # 绘图
        plt.plot(history_data[col_true], label = "实际值", linewidth="1.5")
        plt.plot(cv_plot_df[col_pred],   label = "预测值", linewidth="1.5", ls = "--")
        # plt.plot(predict_data[col_hist_pred], label = "未来预测值", linewidth="1.5")
        # for cutoff in cv_plot_df["cutoff"].unique():
        #     plt.axvline(x = cutoff, color = "red", ls = ":", linewidth="1.0")
        # 图像美化 
        if data_type is None:
            plt.title(f"预测时序图")
        if data_type == "split":
            plt.title(f"配电室 {col_pred.split('-')[0]} 变压器低压进线柜 {col_pred.split('-')[1]} {col_pred.split('-')[2][-1]} 相功率预测时序图")
        elif data_type == "total":
            plt.title(f"配电室 {col_pred.split('-')[0]} 变压器低压进线柜 {col_pred.split('-')[1]} 总功率预测时序图")
        plt.xlabel("Date [h]")
        plt.ylabel("Power [kW]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show();


def plot_cv_predictions_total(cv_plot_df, device_data, data_type = "total"):
    # history ture data
    history_data = device_data["history_data"].dropna()
    history_data.set_index("ds", inplace=True)
    history_data = history_data.loc[history_data.index >= min(cv_plot_df.index), ]
    history_data["line_a"] = history_data["201-linea"] + history_data["203-linea"]
    history_data["line_b"] = history_data["202-lineb"] + history_data["204-lineb"]
    history_data["total"] = history_data["line_a"] + history_data["line_b"]
    # windows pred data
    cv_plot_df["line_a"] = cv_plot_df["201-2AN2a1-Y_pred"] + cv_plot_df["203-2AN1a1-Y_pred"]
    cv_plot_df["line_b"] = cv_plot_df["201-2AN2a1-Y_pred"] + cv_plot_df["203-2AN1a1-Y_pred"]
    cv_plot_df["total"] = cv_plot_df["line_a"] + cv_plot_df["line_b"]
    # future predict data
    predict_data = device_data["predict_data"].dropna()
    predict_data.set_index("ds", inplace=True)
    # plot
    for col_true, col_pred, title_str in zip(["line_a", "line_b", "total"], ["line_a", "line_b", "total"], ["A 路", "B 路", "总"]):
        # 画布
        plt.figure(figsize = (15, 5))
        # 绘图
        plt.plot(history_data[col_true], label = "实际值", linewidth="1.5")
        plt.plot(cv_plot_df[col_pred],   label = "预测值", linewidth="1.5", ls = "--")
        # plt.plot(predict_data[col_hist_pred], label = "未来预测值", linewidth="1.5")
        # for cutoff in cv_plot_df["cutoff"].unique():
        #     plt.axvline(x = cutoff, color = "red", ls = ":", linewidth="1.0")
        # 图像美化 
        if data_type is None:
            plt.title(f"预测时序图")
        elif data_type == "total":
            plt.title(f"配电室变压器低压进线柜{title_str}功率预测时序图")
        plt.xlabel("Date [h]")
        plt.ylabel("Power [kW]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
