# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['AirPassengers', 'AirPassengersDF', 'unique_id', 'ds', 'y', 'AirPassengersPanel', 'snaive', 'airline1_dummy',
           'airline2_dummy', 'AirPassengersStatic', 'generate_series', 'TimeFeature', 'SecondOfMinute', 'MinuteOfHour',
           'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear',
           'time_features_from_frequency_str', 'augment_calendar_df', 'get_indexer_raise_missing',
           'PredictionIntervals', 'add_conformal_distribution_intervals', 'add_conformal_error_intervals',
           'get_prediction_interval_method']

# %% ../nbs/utils.ipynb 3
import random
from itertools import chain
from typing import List, Union
from utilsforecast.compat import DFType

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp

# %% ../nbs/utils.ipynb 6
def generate_series(
    n_series: int,
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 500,
    n_temporal_features: int = 0,
    n_static_features: int = 0,
    equal_ends: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_temporal_features > 0`, then each serie gets temporal features with random values.
    If `n_static_features > 0`, then a static dataframe is returned along the temporal dataframe.
    If `equal_ends == True` then all series end at the same date.

    **Parameters:**<br>
    `n_series`: int, number of series for synthetic panel.<br>
    `min_length`: int, minimal length of synthetic panel's series.<br>
    `max_length`: int, minimal length of synthetic panel's series.<br>
    `n_temporal_features`: int, default=0, number of temporal exogenous variables for synthetic panel's series.<br>
    `n_static_features`: int, default=0, number of static exogenous variables for synthetic panel's series.<br>
    `equal_ends`: bool, if True, series finish in the same date stamp `ds`.<br>
    `freq`: str, frequency of the data, [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).<br>

    **Returns:**<br>
    `freq`: pandas.DataFrame, synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    seasonalities = {"D": 7, "M": 12}
    season = seasonalities[freq]

    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    dates = pd.date_range("2000-01-01", periods=max_length, freq=freq).values
    uids = [np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]

    y = np.arange(total_length) % season + rng.rand(total_length) * 0.5
    temporal_df = pd.DataFrame(
        dict(unique_id=chain.from_iterable(uids), ds=chain.from_iterable(ds), y=y)
    )

    random.seed(seed)
    for i in range(n_temporal_features):
        random.seed(seed)
        temporal_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        temporal_df[f"temporal_{i}"] = np.hstack(temporal_values)
        temporal_df[f"temporal_{i}"] = temporal_df[f"temporal_{i}"].astype("category")
        if i == 0:
            temporal_df["y"] = temporal_df["y"] * (
                1 + temporal_df[f"temporal_{i}"].cat.codes
            )

    temporal_df["unique_id"] = temporal_df["unique_id"].astype("category")
    temporal_df["unique_id"] = temporal_df["unique_id"].cat.as_ordered()

    if n_static_features > 0:
        static_features = np.random.uniform(
            low=0.0, high=1.0, size=(n_series, n_static_features)
        )
        static_df = pd.DataFrame.from_records(
            static_features, columns=[f"static_{i}" for i in range(n_static_features)]
        )

        static_df["unique_id"] = np.arange(n_series)
        static_df["unique_id"] = static_df["unique_id"].astype("category")
        static_df["unique_id"] = static_df["unique_id"].cat.as_ordered()

        return temporal_df, static_df

    return temporal_df

# %% ../nbs/utils.ipynb 12
AirPassengers = np.array(
    [
        112.0,
        118.0,
        132.0,
        129.0,
        121.0,
        135.0,
        148.0,
        148.0,
        136.0,
        119.0,
        104.0,
        118.0,
        115.0,
        126.0,
        141.0,
        135.0,
        125.0,
        149.0,
        170.0,
        170.0,
        158.0,
        133.0,
        114.0,
        140.0,
        145.0,
        150.0,
        178.0,
        163.0,
        172.0,
        178.0,
        199.0,
        199.0,
        184.0,
        162.0,
        146.0,
        166.0,
        171.0,
        180.0,
        193.0,
        181.0,
        183.0,
        218.0,
        230.0,
        242.0,
        209.0,
        191.0,
        172.0,
        194.0,
        196.0,
        196.0,
        236.0,
        235.0,
        229.0,
        243.0,
        264.0,
        272.0,
        237.0,
        211.0,
        180.0,
        201.0,
        204.0,
        188.0,
        235.0,
        227.0,
        234.0,
        264.0,
        302.0,
        293.0,
        259.0,
        229.0,
        203.0,
        229.0,
        242.0,
        233.0,
        267.0,
        269.0,
        270.0,
        315.0,
        364.0,
        347.0,
        312.0,
        274.0,
        237.0,
        278.0,
        284.0,
        277.0,
        317.0,
        313.0,
        318.0,
        374.0,
        413.0,
        405.0,
        355.0,
        306.0,
        271.0,
        306.0,
        315.0,
        301.0,
        356.0,
        348.0,
        355.0,
        422.0,
        465.0,
        467.0,
        404.0,
        347.0,
        305.0,
        336.0,
        340.0,
        318.0,
        362.0,
        348.0,
        363.0,
        435.0,
        491.0,
        505.0,
        404.0,
        359.0,
        310.0,
        337.0,
        360.0,
        342.0,
        406.0,
        396.0,
        420.0,
        472.0,
        548.0,
        559.0,
        463.0,
        407.0,
        362.0,
        405.0,
        417.0,
        391.0,
        419.0,
        461.0,
        472.0,
        535.0,
        622.0,
        606.0,
        508.0,
        461.0,
        390.0,
        432.0,
    ],
    dtype=np.float32,
)

# %% ../nbs/utils.ipynb 13
AirPassengersDF = pd.DataFrame(
    {
        "unique_id": np.ones(len(AirPassengers)),
        "ds": pd.date_range(
            start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
        ),
        "y": AirPassengers,
    }
)

# %% ../nbs/utils.ipynb 20
# Declare Panel Data
unique_id = np.concatenate(
    [["Airline1"] * len(AirPassengers), ["Airline2"] * len(AirPassengers)]
)
ds = np.tile(
    pd.date_range(
        start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
    ).to_numpy(),
    2,
)
y = np.concatenate([AirPassengers, AirPassengers + 300])

AirPassengersPanel = pd.DataFrame({"unique_id": unique_id, "ds": ds, "y": y})

# For future exogenous variables
# Declare SeasonalNaive12 and fill first 12 values with y
snaive = (
    AirPassengersPanel.groupby("unique_id")["y"]
    .shift(periods=12)
    .reset_index(drop=True)
)
AirPassengersPanel["trend"] = range(len(AirPassengersPanel))
AirPassengersPanel["y_[lag12]"] = snaive.fillna(AirPassengersPanel["y"])

# Declare Static Data
unique_id = np.array(["Airline1", "Airline2"])
airline1_dummy = [0, 1]
airline2_dummy = [1, 0]
AirPassengersStatic = pd.DataFrame(
    {"unique_id": unique_id, "airline1": airline1_dummy, "airline2": airline2_dummy}
)

AirPassengersPanel.groupby("unique_id").tail(4)

# %% ../nbs/utils.ipynb 26
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex):
        return print("Overwrite with corresponding feature")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    if freq_str not in ["Q", "M", "MS", "W", "D", "B", "H", "T", "S"]:
        raise Exception("Frequency not supported")

    if freq_str in ["Q", "M", "MS"]:
        return [cls() for cls in [MonthOfYear]]
    elif freq_str == "W":
        return [cls() for cls in [DayOfMonth, WeekOfYear]]
    elif freq_str in ["D", "B"]:
        return [cls() for cls in [DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "H":
        return [cls() for cls in [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "T":
        return [
            cls() for cls in [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        ]
    else:
        return [
            cls()
            for cls in [
                SecondOfMinute,
                MinuteOfHour,
                HourOfDay,
                DayOfWeek,
                DayOfMonth,
                DayOfYear,
            ]
        ]


def augment_calendar_df(df, freq="H"):
    """
    > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    df = df.copy()

    freq_map = {
        "Q": ["month"],
        "M": ["month"],
        "MS": ["month"],
        "W": ["monthday", "yearweek"],
        "D": ["weekday", "monthday", "yearday"],
        "B": ["weekday", "monthday", "yearday"],
        "H": ["dayhour", "weekday", "monthday", "yearday"],
        "T": ["hourminute", "dayhour", "weekday", "monthday", "yearday"],
        "S": [
            "minutesecond",
            "hourminute",
            "dayhour",
            "weekday",
            "monthday",
            "yearday",
        ],
    }

    ds_col = pd.to_datetime(df.ds.values)
    ds_data = np.vstack(
        [feat(ds_col) for feat in time_features_from_frequency_str(freq)]
    ).transpose(1, 0)
    ds_data = pd.DataFrame(ds_data, columns=freq_map[freq])

    return pd.concat([df, ds_data], axis=1), freq_map[freq]

# %% ../nbs/utils.ipynb 29
def get_indexer_raise_missing(idx: pd.Index, vals: List[str]) -> List[int]:
    idxs = idx.get_indexer(vals)
    missing = [v for i, v in zip(idxs, vals) if i == -1]
    if missing:
        raise ValueError(f"The following values are missing from the index: {missing}")
    return idxs

# %% ../nbs/utils.ipynb 31
class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        method: str = "conformal_distribution",
    ):
        """
        n_windows : int
            Number of windows to evaluate.
        method : str, default is conformal_distribution
            One of the supported methods for the computation of prediction intervals:
            conformal_error or conformal_distribution
        """
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.method = method

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, method='{self.method}')"
        )

# %% ../nbs/utils.ipynb 32
def add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    n_series: int,
    horizon: int,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
        scores = scores.transpose(1, 0, 2)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)
    return fcst_df

# %% ../nbs/utils.ipynb 33
def add_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    n_series: int,
    horizon: int,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
        scores = scores.transpose(1, 0, 2)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df

# %% ../nbs/utils.ipynb 34
def get_prediction_interval_method(method: str):
    available_methods = {
        "conformal_distribution": add_conformal_distribution_intervals,
        "conformal_error": add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f'please choose one of {", ".join(available_methods.keys())}'
        )
    return available_methods[method]