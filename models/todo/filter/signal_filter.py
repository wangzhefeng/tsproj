# -*- coding: utf-8 -*-


# ***************************************************
# * File        : single_filter.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-17
# * Version     : 0.1.071723
# * Description : 滤波算法
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy import signal


def limiting_filter(inputs, per):
    """
    限幅滤波(程序判断滤波法)
    Args:
        inputs:
        per:
    """
    pass


def median_filter(inputs, per):
    """
    中位值滤波
    Args:
        inputs:
        per:
    """
    pass


def arithmetic_average_filter(inputs, per):
    '''
    算术平均滤波法
    Args:
        inputs:
        per:
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


def sliding_average_filter(inputs, per):
    '''
    递推平均滤波法(滑动平均滤波)
    Args:
        inputs:
        per: 
    filter = np.ones(200)*(1/200)
    sample_filter_03 = np.convolve(data,filter,'valid')
    sample_filter_012 = np.convolve(data,filter,'valid')
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean+tmp.mean())/2)
        tmpmean = tmp.mean()
    return mean


def median_average_filter(inputs, per):
    '''
    中位值平均滤波法(防脉冲干扰平均滤波)
    Args:
        inputs:
        per:
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp,np.where(tmp==tmp.max())[0],axis = 0)
        tmp = np.delete(tmp,np.where(tmp==tmp.min())[0],axis = 0)
        mean.append(tmp.mean())
    return mean


def amplitude_limiting_average_filter(inputs, per, amplitude):
    '''
    限幅平均滤波法
    Args:
        inputs:
        per:
        amplitude: 限制最大振幅
    '''
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
            inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
    inputs = inputs.reshape((-1,per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0] #上一次限幅后结果
    for tmp in inputs:
        for index,newtmp in enumerate(tmp):
            if np.abs(tmpnum-newtmp) > amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean+tmp.mean())/2)
        tmpmean = tmp.mean()
    return mean


def first_order_lag_filter(inputs, a):
    '''
    一阶滞后滤波法
    Args:
        inputs:
        a: 滞后程度决定因子, 0~1
    '''
    tmpnum = inputs[0] #上一次滤波结果
    for index,tmp in enumerate(inputs):
        inputs[index] = (1-a)*tmp + a*tmpnum
        tmpnum = tmp
    return inputs
 

def weight_backstep_average_filter(inputs, per):
    '''
    加权递推平均滤波法
    Args:
        inputs: 
        per:
    '''
    weight = np.array(range(1,np.shape(inputs)[0]+1)) # 权值列表
    weight = weight/weight.sum()

    for index,tmp in enumerate(inputs):
        inputs[index] = inputs[index]*weight[index]
    return inputs


def shake_off_filter(inputs, N):
    '''
    消抖滤波法
    Args:
        inputs:
        N: 消抖上限
    '''
    usenum = inputs[0] #有效值
    i = 0 # 标记计数器
    for index,tmp in enumerate(inputs):
        if tmp != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


def amplitude_limiting_shake_off_filter(inputs, amplitude, N):
    '''
    限幅消抖滤波法
    Args:
        inputs:
        amplitude: 限制最大振幅
        N:         消抖上限
    '''
    tmpnum = inputs[0]
    for index,newtmp in enumerate(inputs):
        if np.abs(tmpnum-newtmp) > amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    usenum = inputs[0]
    i = 0
    for index2,tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    return inputs


def low_pass_filer(data, N, Wn):
    """
    低通滤波:  低通滤波指的是去除高于某一阈值频率的信号

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母
    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "lowpass")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def high_pass_filter(data, N, Wn):
    """
    高通滤波: 高通滤波去除低于某一频率的信号

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母

    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "highpass")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def band_pass_filter(data, N, Wn):
    """
    带通滤波: 带通滤波指的是类似低通高通的结合保留中间频率信号

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母

    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "bandpass")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data


def band_stop_filter(data, N, Wn):
    """
    带阻滤波: 带阻滤波也是低通高通的结合只是过滤掉的是中间部分

    Args:
        data ([type]): 要过滤的信号
        N ([type]): 滤波器的阶数
        Wn ([type]): 归一化截止频率, Wn = 2 * 截止频率 / 采样频率
            - 根据采样定理, 采样频率要大于两倍的信号本身最大的频率, 才能还原信号. 
            - 截止频率一定小于信号本身最大的频率, 所以 Wn 一定在 0 和 1 之间
            - 当构造带通滤波器或者带阻滤波器时, Wn为长度为2的列表
        b: 滤波器的分子
        a: 滤波器的分母

    Returns:
        [type]: [description]
    """
    b, a = signal.butter(N = N, Wn = Wn, btype = "bandstop")
    filted_data = signal.filtfilt(b, a, data)
    return filted_data




# 测试代码 main 函数
def main():
    num = signal.chirp(np.arange(0, 0.5, 1 / 4410.0), f0 = 10, t1 = 0.5, f1 = 1000.0)
    result = arithmetic_average_filter(num.copy(), 30)
    plt.subplot(2, 1, 1)
    plt.plot(num)
    plt.subplot(2, 1, 2)
    plt.plot(result)
    plt.show()

if __name__ == "__main__":
    main()

