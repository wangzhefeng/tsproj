# -*- coding: utf-8 -*-


# ***************************************************
# * File        : wavelet_analysis.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-02
# * Version     : 0.1.110200
# * Description : Wavelet Analysis 时间序列小波分析
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
import pandas as pd

# import DataAPI
import pywt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA


# TODO
def wave_analysis(data, start, end):
    """
    基于小波变换的时间序列预测
    思路: 
        将数据序列进行小波分解, 每一层分解的结果是上次分解得到的低频信号再分解成低频和高频两个部分. 如此进过N层分解后源信号X被分解为: 
        X = D1 + D2 + ... + DN + AN 其中D1,D2,...,DN分别为第一层、第二层到等N层分解得到的高频信号, AN为第N层分解得到的低频信号. 
        方案为对D1,D2...DN和AN分别进行预测, 然后进行小波重构实现对源信号的预测. 
    步骤: 
    (1)对原序列进行小波分解, 得到各层小波系数; 
    (2)对各层小波系数分别建立 ARMA 模型, 对各层小波系数进行预测; 
    (3)用得到的预测小波系数重构数据
    """
    # --------
    # 数据读取
    # --------
    # data = DataAPI.MktIdxdGet(ticker = "000001", beginDate = start, endDate = end, field = ["tradeDate", "closeIndex"], pandas = "1")
    # 训练数据
    train_index = np.array(data["closeIndex"])[:-10]
    train_date_list = np.array(data["tradeDate"])[:-10]
    # 测试数据
    test_index = np.array(data["closeIndex"])[-10:]
    test_date_list = np.array(data["tradeDate"])[-10:]
    # --------
    # 时间序列数据分解
    # --------
    # 分解得到第4层低频部分系数和全部4层高频部分系数
    A2, D2, D1 = pywt.wavedec(train_index, "db4", mode = "sym", level = 2)
    coeff = [A2, D2, D1]
    # --------
    # 对各层小波系数求解模型系数
    # --------
    order_A2 = sm.tsa.arma_order_select_ic(A2, ic = "aic")["aic_min_order"] # AIC 准侧求解模型阶数 p, q
    order_D2 = sm.tsa.arma_order_select_ic(D2, ic = "aic")["aic_min_order"] # AIC 准侧求解模型阶数 p, q
    order_D1 = sm.tsa.arma_order_select_ic(D2, ic = "aic")["aic_min_order"] # AIC 准侧求解模型阶数 p, q
    # --------
    # 对各层小波系数构建 ARMA 模型
    # --------
    model_A2 = ARMA(A2, order = order_A2)
    model_D2 = ARMA(D2, order = order_D2)
    model_D1 = ARMA(D1, order = order_D1)
    result_A2 = model_A2.fit()
    result_D2 = model_D2.fit()
    result_D1 = model_D1.fit()
    # 对所有序列分解
    A2_all, D2_all, D1_all = pywt.wavedec(np.array(data["closeIndex"]), "db4", mode = "sym", level = 2)
    # 求出差值, 则delta序列对应的是每层小波系数ARMA模型需要预测的步数
    delta = [
        len(A2_all) - len(A2), 
        len(D2_all) - len(D2),
        len(D1_all) - len(D1), 
    ]
    # 预测小波系数 包括 in-sample 的和 out-sample的需要预测的小波系数
    pA2 = model_A2.predict(params = result_A2.params, start = 1, end = len(A2) + delta[0])
    pD2 = model_D2.predict(params = result_D2.params, start = 1, end = len(D2) + delta[1])
    pD1 = model_D1.predict(params = result_D1.params, start = 1, end = len(D1) + delta[2])
    # 重构
    coeff_new = [pA2, pD2, pD1]
    denoised_index = pywt.waverec(coeff_new, "db4")
    # 输出预测值
    temp_data_wt = {
        "real_value": test_index,
        "pred_value_wt": denoised_index[-10:],
        "err_wt": denoised_index[-10:] - test_index,
        "err_rate_wt/%": (denoised_index[-10:] - test_index) / test_index * 100
    }
    predict_wt = pd.DataFrame(
        temp_data_wt, 
        index = test_date_list, 
        columns = [
            "real_value",
            "pred_value_wt",
            "err_wt",
            "err_rate_wt/%",
        ]
    )
    print(predict_wt)
    all_data = DataAPI.MktIdxdGet(ticker = "000001", beginDate = start, endDate = end, field = ["tradeDate", "closeIndex"], pandas = "1")
    all_data = all_data.set_index(all_data["tradeDate"])
    all_data.plot(figsize = (25, 10))





# 测试代码 main 函数
def main():
    wave_analysis("20110101", "20120531")

if __name__ == "__main__":
    main()

