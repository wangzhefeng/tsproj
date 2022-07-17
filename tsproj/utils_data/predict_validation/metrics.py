# -*- coding: utf-8 -*-
# ! /usr/bin/env python3

# *********************************************
# * Author      : canping Chen, Yuepeng Zheng
# * Email       : canping.chen@yo-i.net, yuepeng.zheng@yo-i.net
# * Date        : 2021.11.19
# * Description : 预测结果评价
# * Link        :
# **********************************************
import numpy as np
import time
import datetime
import pandas as pd
import logging
from scipy import interpolate


def RSE(pred, true):
    """
    函数功能: 计算预测值的相对平方误差(Relative Squared Error). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    函数功能: 计算预测值的相关系数(Correlation). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 *
                (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean()


def MAE(pred, true):
    """
    函数功能: 计算预测值的平均绝对误差(Mean Ablolute Error). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    函数功能: 计算预测值的均方误差(Mean Squared Error). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    函数功能: 计算预测值的均方根误差(Root Mean Square Error,). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    函数功能: 计算预测值的平均绝对百分比误差(Mean Ablolute Percentage Error). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    if 0 in true:
        pred = pd.Series(pred)
        true = pd.Series(true)
        pred = pred[true != 0]
        true = true[true != 0]
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    函数功能: 计算预测值的平均平方百分比误差(Mean Scaled Percentage Error). 
    Parameters:
        pred:预测结果
    Return: 
        true:实际值
    """
    if 0 in true:
        pred = pd.Series(pred)
        true = pd.Series(true)
        pred = pred[true != 0]
        true = true[true != 0]
    return np.mean(np.square((pred - true) / true))


def PredictMetrics(pred, true):
    """
    函数功能: 计算预测值的各项评价指标, 包括RSE,CORR,MAE,MSE,RMSE,MAPE,MSPE
    Parameter: 
        pred:预测结果,一维数组
        true:实际值, 一维数组
    Return:
        valid_data:pandas dataframe [timestamp, pred1, pred2, ... true]/[时间戳, 预测值1, 预测值2...实际值]
    """
    if len(pred) == 0 or len(true) == 0:
        return None
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return {
        "rse": rse,
        "corr": corr,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe}


def PredictValidation(real_dt, predict_dt, p_interval, validate_idx=[-1], interp_kind='linear'):
    """
    Parameter: 
        real_dt: pandas dataframe [timestamp, true_param], 第一列为时间戳
        predict_dt: pandas dataframe[timestamp,  predict_t1, predict_t2, ..., predict_tn], 第一列为时间戳
        p_interval: predict interval: 预测值的时间间隔, 精确到秒
        validate_idx: validate index: list, int, 所要评价的预测值索引,ie.[1,5,15...]
        interp_kind: 插值类型, 支持的类型包括"linear", 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero', 默认为'linear'
    Return:
        result: dict, {validate_index[1]:pandas dataframe[timestamp,predict,true_param],validate_index[5]:pandas dataframe...}
    """
    assert isinstance(real_dt, pd.DataFrame), 'real_dt应为Pandas DataFrame格式'
    assert isinstance(
        predict_dt, pd.DataFrame), 'predict_dt应为Pandas DataFrame格式'

    result = {}

    for i in validate_idx:
        try:
            assert i <= len(
                predict_dt.columns)-1, f'validate_idx({i})超出范围, 应小于等于{len(predict_dt.columns)-1}'
            assert isinstance(
                i, int) and i > 0, f'validate_idx({i})超出范围, 应为正整数'
        except AssertionError as e:
            logging.error(e)
            continue

        _predict_dt = predict_dt.iloc[:, [0, i]].copy()  # 选取需要评价的预测值
        _predict_dt.columns = ['timestamp', 'pred']
        _real_dt = real_dt.copy()
        _real_dt.columns = ['timestamp', 'true']

        time_space = p_interval*i  # 预测提前量(秒)
        _predict_dt['timestamp'] += time_space  # 预测结果时间戳
        _predict_dt = _predict_dt[(_predict_dt['timestamp'] >= min(_real_dt['timestamp'])) & (
            _predict_dt['timestamp'] <= max(_real_dt['timestamp']))]  # 为了对实际值按照预测时间戳进行插值处理, 先使预测结果时间戳缩小至实际值时间戳区间
        f = interpolate.interp1d(
            _real_dt['timestamp'], _real_dt['true'], kind=interp_kind)
        # 通过插值方法取得的预测结果时间戳对应的实际值数据
        _predict_dt['true'] = f(_predict_dt['timestamp'])
        result.update({i: _predict_dt})

    return result


def main():
    # 测试
    import random
    timestamp = np.arange(time.time()-1100, time.time())
    rdm = np.random.rand(1000)*1000
    rdm_prd = np.random.rand(1100)*1000
    real_dt = pd.DataFrame(
        {'timestamp': timestamp[100:] + np.random.random(1000)/10, 'value': rdm})
    predict_dt = {}
    predict_dt.update({'timestamp': timestamp + np.random.random(1100)/10})
    for i in range(10):
        j = f'value{i}'
        predict_dt.update({j: rdm_prd})
    predict_dt = pd.DataFrame(predict_dt)
    p_interval = 60  # seconds

    # 检查数据有效性, 输出以validate_idx为索引的有效数据
    valid_data = PredictValidation(real_dt, predict_dt,
                                   p_interval, validate_idx=[1, 3, 5, 15, 20, 100, 0, -1, -2])
    print('validate_idx:', valid_data.keys())
    # 计算评价指标
    for i in valid_data:
        print(f'validate_idx[{i}]:', PredictMetrics(
            valid_data[i]['pred'], valid_data[i]['true']))


if __name__ == '__main__':
    main()
