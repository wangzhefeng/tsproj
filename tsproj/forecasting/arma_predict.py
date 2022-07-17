from mimetypes import init
from statistics import mode
import sys
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller as ADF


class AutoARMA:
    def __init__(self, ts) -> None:
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.p = None
        self.q = None
        self.properModel = None
        self.diff_count = 0
        self.bic = sys.maxsize

    def Predict(self, pred_len=20):
        # 样本内预测
        predict_inside = deepcopy(self.predict_ts)
        predict_outside = self.properModel.predict(len(predict_inside), len(predict_inside) + pred_len, dynamic=True)
        init_value = self.data_ts[0]
        for i in range(self.diff_count):
            predict_inside = predict_inside.cumsum()  # 差分还原
            predict_inside = init_value + predict_inside
            predict_inside = [init_value] + list(predict_inside)
            init_value = list(predict_inside)[0]

            start_pre_value = list(predict_inside)[-1]
            predict_outside = predict_outside.cumsum()  # 差分还原
            predict_outside = start_pre_value + predict_outside
            predict_outside = [start_pre_value] + list(predict_outside)
        return predict_inside, predict_outside

    def Train(self):
        train_ts, p_q = self.Confirm_p_q()
        init_p, init_q = p_q
        max_p = init_p + 2
        max_q = init_q + 2
        for p in np.arange(max_p):
            for q in np.arange(max_q):
                # print p,q,self.bic
                model = ARMA(train_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1)
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.bic = bic
                    self.properModel = results_ARMA
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    def Confirm_p_q(self):
        train_ts = self.TestSteady()
        p_max = int(len(train_ts) / 10)
        q_max = int(len(train_ts) / 10)
        AIC = sm.tsa.arma_order_select_ic(train_ts, max_ar=p_max, max_ma=q_max, ic='aic')['aic_min_order']
        BIC = sm.tsa.arma_order_select_ic(train_ts, max_ar=p_max, max_ma=q_max, ic='bic')['bic_min_order']
        HQIC = sm.tsa.arma_order_select_ic(train_ts, max_ar=p_max, max_ma=q_max, ic='hqic')['hqic_min_order']
        return train_ts, AIC

    def TestSteady(self, p_val_eps=1E-2):
        ts_data = deepcopy(self.data_ts)
        res_ADF = ADF(ts_data)
        p_value = res_ADF[1]
        self.diff_count = 0
        while p_value > p_val_eps:
            self.diff_count += 1
            ts_data = ts_data.diff(1).dropna()
            res_ADF = ADF(ts_data)
            p_value = res_ADF[1]
        return ts_data


if __name__ == "__main__":
    yv = np.array([2800, 2811, 2832, 2850, 2880, 2910,
                   2960, 3023, 3039, 3056, 3138, 3150, 3198, 3100, 3029,
                   2950, 2989, 3012, 3050, 3142, 3252, 3342, 3365, 3385,
                   3340, 3410, 3443, 3428, 3554, 3615, 3646, 3614, 3574,
                   3635, 3738, 3764, 3788, 3820, 3840, 3875, 3900, 3942,
                   4000, 4021, 4055])
    yv_series = pd.Series(yv)
    model = AutoARMA(yv_series)
    model.Train()

    print(model.p, model.q, model.bic, model.diff_count)
    pred_inside, pred_outside = model.Predict()

    print(len(pred_inside), len(yv))
    plt.plot(model.data_ts)
    plt.plot(pred_inside)
    plt.show()
