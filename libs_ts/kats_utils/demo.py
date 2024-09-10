# %%
import sys
sys.path.append("../../datasets/kats_dataset/")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from kats.consts import TimeSeriesData

# %% [markdown]
# # 1.数据

# %%
air_passengers_df = pd.read_csv("../../datasets/kats_dataset/air_passengers.csv")
air_passengers_df.columns = ["time", "value"]
air_passengers_df.head()

# %%
multi_ts_df = pd.read_csv("../../datasets/kats_dataset/multi_ts.csv", index_col = 0)
multi_ts_df.columns = ["time", "v1", "v2"]
multi_ts_df.head()

# %% [markdown]
# # 2.TimeSeriesData 建立

# %%
air_passengers_ts = TimeSeriesData(air_passengers_df)
multi_ts = TimeSeriesData(multi_ts_df)
print(type(air_passengers_ts))
print(type(multi_ts))
print(type(air_passengers_ts.time))
print(type(air_passengers_ts.value))
print(type(multi_ts.time))
print(type(multi_ts.value))

# %%
air_passengers_ts_from_series = TimeSeriesData(time = air_passengers_df.time, value = air_passengers_df.value)
multi_ts_from_series = TimeSeriesData(time = multi_ts_df.time, value = multi_ts_df[["v1", "v2"]])
print(air_passengers_ts_from_series)
print(multi_ts_from_series)

# %%
air_passengers_ts_unixtime = air_passengers_df.time.apply(lambda x: datetime.timestamp(parser.parse(x)))
print(air_passengers_ts_unixtime)
ts_from_unixtime = TimeSeriesData(
    time = air_passengers_ts_unixtime,
    value = air_passengers_df.value,
    use_unix_time = True,
    unix_time_units = "s"
)
ts_from_unixtime

# %% [markdown]
# # 3.TimeSeriesData 操作

# %%
air_passengers_ts[1:5]

# %%
air_passengers_ts[1:5] + air_passengers_ts[1:5]

# %%
multi_ts == multi_ts_from_series

# %%
len(air_passengers_ts)

# %%
ts_1 = air_passengers_ts[0:3]
ts_2 = air_passengers_ts[3:7]
ts_1.extend(ts_2)
ts_1

# %%
air_passengers_ts.plot(cols = ["value"])
plt.show()

# %%
multi_ts.plot(cols = ["v1", "v2"])
plt.show()

# %%
air_passengers_ts.to_dataframe().head()

# %%
air_passengers_ts.to_array()[0:5]

# %%
air_passengers_ts.is_empty()

# %%
air_passengers_ts.is_univariate()

# %%
multi_ts.is_univariate()

# %% [markdown]
# # 4.时间序列预测

# %% [markdown]
# ## 4.1 Linear

# %%


# %% [markdown]
# ## 4.2 Quadratic

# %%


# %% [markdown]
# ## 4.3 ARIMA

# %%


# %% [markdown]
# ## 4.4 SARIMA

# %%
from kats.models.sarima import SARIMAModel, SARIMAParams

params = SARIMAParams(
    p = 2,
    d = 1,
    q = 1,
    trend = "ct",
    seasonal_order = (1, 0, 1, 12)
)
sarima_model = SARIMAModel(data = air_passengers_ts, params = params)
sarima_model.fit()
fcst = sarima_model.predict(steps = 30, freq = "MS")

# %%
fcst.head()

# %%
sarima_model.plot()

# %% [markdown]
# ## 4.5 Holt-Winters

# %%
from kats.models.holtwinters import HoltWintersModel, HoltWintersParams

params = HoltWintersParams(
    trend  = "add",
    # damped = False,
    seasonal = "mul",
    seasonal_periods = 12,
)
holtwinters_model = HoltWintersModel(data = air_passengers_ts, params = params)
holtwinters_model.fit()
fcst = holtwinters_model.predict(steps = 30, alpha = 0.1)

# %%
fcst.head()

# %%
holtwinters_model.plot()

# %% [markdown]
# ## 4.6 Prophet

# %%
from kats.models.prophet import ProphetModel, ProphetParams

params = ProphetParams(seasonality_mode = "multiplicative")
prophet_model = ProphetModel(air_passengers_ts, params)
prophet_model.fit()
fcts = prophet_model.predict(steps = 30, freq = "MS")

# %%
fcts.head()

# %%
prophet_model.plot()

# %% [markdown]
# ## 4.7 AR-Net

# %%


# %% [markdown]
# ## 4.8 LSTM

# %%


# %% [markdown]
# ## 4.9 Theta

# %%
from kats.models.theta import ThetaModel, ThetaParams

params = ThetaParams(m = 12)
theta_model = ThetaModel(data = air_passengers_ts, params = params)
theta_model.fit()
fcst = theta_model.predict(steps = 30, alpha = 0.2)

# %%
fcst.head()

# %%
theta_model.plot()

# %% [markdown]
# ## 4.10 VAR--Multivariate Model

# %%
multi_ts

# %%
from kats.models.var import VARModel, VARParams

params = VARParams()
var_model = VARModel(multi_ts, params)
var_model.fit()
fcst = var_model.predict(steps = 90)
var_model.plot()

# %% [markdown]
# ## 4.11 Ensemble

# %%
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models import (
    arima, 
    holtwinters, 
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)

model_params = EnsembleParams([
    BaseModelParams("arima", arima.ARIMAParams(p = 1, d = 1, q = 1)),
    BaseModelParams("sarima", sarima.SARIMAParams(p = 2, d = 1, q = 1, trend = "ct", seasonal_order = (1, 0, 1, 12), enforce_invertibility = False, enforce_stationarity = False)),
    BaseModelParams("prophet", prophet.ProphetParams()),
    BaseModelParams("linear", linear_model.LinearModelParams()),
    BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
    BaseModelParams("theta", theta.ThetaParams(m = 12)),
])
KatsEnsembleParam = {
    "models": model_params,
    "aggregation": "median",
    "seasonality_length": 12,
    "decomposition_method": "multiplicative",
}

kat_ensemble_model = KatsEnsemble(data = air_passengers_ts, params = KatsEnsembleParam)
kat_ensemble_model.fit()
fcst = kat_ensemble_model.predict(steps = 30)
kat_ensemble_model.aggregate()
kat_ensemble_model.plot()

# %% [markdown]
# ## 4.12 超参数调参

# %%
import kats.utils.time_series_parameter_tuning as tpt
from kats.consts import ModelEnum, SearchMethodEnum
from kats.models.arima import ARIMAModel, ARIMAParams
# from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
# from ax.models.random.sobol import SobolGenerator
# from ax.models.random.uniform import UniformGenerator

# train and test data
split = int(0.8 * len(air_passengers_df))
train_ts = air_passengers_ts[0:split]
test_ts = air_passengers_ts[split:]

def evaluation_function(params):
    arima_params = ARIMAParams(
        p = params["p"],
        d = params["d"],
        q = params["q"],
    )
    arima_model = ARIMAModel(train_ts, arima_params)
    arima_model.fit()
    arima_model_pred = arima_model.predict(steps = len(test_ts))
    error = np.mean(np.abs(arima_model_pred["fcst"].values - test_ts.value.values))
    return error

# model params tune
parameters_grid_search = [
    {
        "name": "p",
        "type": "choice",
        "values": list(range(1, 3)),
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "d",
        "type": "choice",
        "values": list(range(1, 3)),
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "q",
        "type": "choice",
        "values": list(range(1, 3)),
        "value_type": "int",
        "is_ordered": True,
    }
]
parameter_tuner_grid = tpt.SearchMethodFactory.create_search_method(
    objective_name = "evaluation_metric",
    parameters = parameters_grid_search,
    selected_search_method = SearchMethodEnum.GRID_SEARCH
)
parameter_tuner_grid.generate_evaluate_new_parameter_values(evaluation_function = evaluation_function)
parameter_tunning_results_grid = (parameter_tuner_grid.list_parameter_value_scores())
parameter_tunning_results_grid

# %% [markdown]
# ## 4.13 Backtesting

# %%
from kats.utils.backtesters import BackTesterSimple

ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]
backtester_errors = {}

# %%
from kats.models.arima import ARIMAModel, ARIMAParams

params_arima = ARIMAParams(p = 2, d = 1, q = 1)
backtester_arima = BackTesterSimple(
    error_methods = ALL_ERRORS,
    data = air_passengers_ts,
    params = params,
    train_percentage = 75,
    test_percentage = 25,
    model_class = ARIMAModel
)
backtester_arima.run_backtest()

backtester_errors["arima"] = {}
for error, value in backtester_arima.errors.items():
    backtester_errors["arima"][error] = value

# %%
from kats.utils.backtesters import BackTesterSimple
from kats.models.prophet import ProphetModel, ProphetParams

params_prophet = ProphetParams(seasonality_mode = "multiplicative")
backtester_prophet = BackTesterSimple(
    error_methods = ALL_ERRORS,
    data = air_passengers_ts,
    params = params_prophet,
    train_percentage = 75,
    test_percentage = 25,
    model_class = ProphetModel
)
backtester_prophet.run_backtest()

backtester_errors["prophet"] = {}
for error, value in backtester_prophet.errors.items():
    backtester_errors["prophet"][error] = value

# %%
pd.DataFrame.from_dict(backtester_errors)

# %% [markdown]
# # 5.时间序列异常检测

# %% [markdown]
# ## 5.1 时间序列异常值检测

# %% [markdown]
# ### 5.1.1 单变量异常值检测

# %%
from kats.detectors.outlier import OutlierDetector

# data
air_passengers_outlier_df = air_passengers_df.copy(deep = True)
air_passengers_outlier_df.loc[air_passengers_outlier_df.time == "1950-12-01", "value"] *= 5
air_passengers_outlier_df.loc[air_passengers_outlier_df.time == "1959-12-01", "value"] *= 4
air_passengers_outlier_df.plot(x = "time", y = "value", figsize = (15, 8))
plt.show()

# %%
# 异常值检测
air_passengers_outlier_ts = TimeSeriesData(air_passengers_outlier_df)
ts_outlierDetection = OutlierDetector(air_passengers_outlier_ts, "additive")
ts_outlierDetection.detector()
ts_outlierDetection.outliers[0]

# %%
# 异常值处理
air_passengers_ts_outliers_removed = ts_outlierDetection.remover(interpolate = False)
air_passengers_ts_outliers_interpolated = ts_outlierDetection.remover(interpolate = True)

# %%
fig, ax = plt.subplots(figsize = (30, 8), nrows = 1, ncols = 3)
air_passengers_ts.to_dataframe().plot(x = "time", y = "value", ax = ax[0])
ax[0].set_title("Air Passengers")
air_passengers_ts_outliers_removed.to_dataframe().plot(x = "time", y = "y_0", ax = ax[1])
ax[1].set_title("Outliers Removed: No interpolation")
air_passengers_ts_outliers_interpolated.to_dataframe().plot(x = "time", y = "y_0", ax = ax[2])
ax[2].set_title("Outliers Removed: With interpolation")
plt.show()

# %% [markdown]
# ### 5.1.2 MultivariateAnomaly

# %%


# %% [markdown]
# ## 5.2 时间序列突变点检测

# %%
from kats.consts import TimeSeriesData, TimeSeriesIterator
from kats.detectors.cusum_detection import CUSUMDetector

np.random.seed(10)
df = pd.DataFrame({
    "time": pd.date_range("2019-01-01", "2019-03-01"),
    "increase": np.concatenate([np.random.normal(1, 0.2, 30), np.random.normal(2, 0.2, 30)]),
    "decrease": np.concatenate([np.random.normal(1, 0.3, 50), np.random.normal(0.5, 0.3, 10)]),
})

# %% [markdown]
# ### 5.2.1 CUSUM

# %%
# detect increase
timeseries = TimeSeriesData(df.loc[:, ["time", "increase"]])
detector = CUSUMDetector(timeseries)
change_points = detector.detector(change_directions = ["increase"])

plt.xticks(rotation = 45)
detector.plot(change_points)
plt.show()

# %%
# detect decrease
timeseries = TimeSeriesData(df.loc[:, ["time", "decrease"]])
detector = CUSUMDetector(timeseries)
change_points = detector.detector(change_directions = ["decrease"])

plt.xticks(rotation = 45)
detector.plot(change_points)
plt.show()

# %%
# detect increase
timeseries = TimeSeriesData(df.loc[:, ["time", "increase"]])
detector = CUSUMDetector(timeseries)
change_points = detector.detector(change_directions = ["decrease"])

plt.xticks(rotation = 45)
detector.plot(change_points)
plt.show()

# %%
# detect increase
timeseries = TimeSeriesData(df.loc[:, ["time", "increase"]])
detector = CUSUMDetector(timeseries)
change_points = detector.detector()

plt.xticks(rotation = 45)
detector.plot(change_points)
plt.show()

# %% [markdown]
# ### 5.2.2 BOCP

# %%


# %% [markdown]
# ### 5.2.3 RobustStat

# %%


# %% [markdown]
# ### 5.2.4 Comparing the Changepoint Detectors

# %%


# %% [markdown]
# ## 5.3 时间序列趋势变化检测

# %% [markdown]
# ### 5.3.1 模拟数据

# %%
# basic usage
from kats.utils.simulator import Simulator

sim = Simulator(n = 365, start = "2020-01-01", freq = "D")
tsd = sim.trend_shift_sim(
    noise = 200, 
    seasonal_period = 7, 
    seasonal_magnitude = 0.007, 
    cp_arr = [250], 
    intercept = 10000, 
    trend_arr = [40, -20]
)
tsd.plot(cols = ["value"])

# %% [markdown]
# ### 5.3.2 趋势检测

# %%
from kats.detectors.trend_mk import MKDetector

detector = MKDetector(data = tsd, threshold = 0.8)
detector_time_points = detector.detector(direction = "down", window_size = 20, freq = "weekly")
detector.plot(detector_time_points)

# %% [markdown]
# ### 5.3.3 结果解释

# %%
cp, meta = detector_time_points[0]
print(cp)
print(meta.__dict__)

# %% [markdown]
# # 6.时间序列特征提取

# %%
from kats.tsfeatures.tsfeatures import TsFeatures

tsFeatures = TsFeatures()
features_air_passengers = tsFeatures.transform(air_passengers_ts)
features_air_passengers


