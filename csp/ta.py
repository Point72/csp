import math
import numpy as np
import typing
from datetime import timedelta

import csp
from csp import stats, ts

# from typing import List, TypeVar


__all__ = ["macd", "bollinger", "momentum", "obv"]


# T = TypeVar("T")


"""
Basic TA Lib
"""


@csp.graph
def macd(x: ts[typing.Union[float, np.ndarray]], a: int, b: int) -> ts[typing.Union[float, np.ndarray]]:
    """
    Run a MACD calculation on the given time series.
    MACD = 12 Period EMA - 26 Period EMA

    :param x: the time-series data
    :param a: the slow EMA period
    :param b: the fast EMA period
    """

    fast_ema = stats.ema(x, span=b, min_data_points=b)
    slow_ema = stats.ema(x, span=a, min_data_points=a)
    return fast_ema - slow_ema


@csp.graph
def bollinger(
    x: ts[typing.Union[float, np.ndarray]], a: int, t: typing.Union[int, timedelta]
) -> {str: ts[typing.Union[float, np.ndarray]]}:
    """
    Upper band = 20-day SMA + (20-day SD x 2)
    Middle band = 20-day SMA
    Lower band = 20-day SMA - (20-day SD x 2)

    :param x: the time-series data
    :param a: standard deviation multiplicity
    :param t: the interval over which to calculate an SMA, can either be a tick value (int) or a timedelta.
    """

    sd = stats.stddev(
        x, min_data_points=t, interval=t, min_window=0
    )  # min_window = 0 forces NaN to be generated when the amount of elements in interval is less than `t`, otherwise it does not return an array of shape == `x`
    middle_band = stats.mean(x, interval=t, min_data_points=t, min_window=0)
    upper_band = middle_band + (sd * a)
    lower_band = middle_band - (sd * a)

    return {"middle": middle_band, "upper": upper_band, "lower": lower_band}


@csp.node
def _momentum(x: ts[typing.Union[float, np.ndarray]], n: typing.Union[int, timedelta]) -> ts[float]:
    with csp.state():
        csp.set_buffering_policy(x, tick_count=n + 1)
    if csp.ticked(x):
        price_n_idx_ago = csp.values_at(x, -(n), None)
        if len(price_n_idx_ago) == n + 1:
            return price_n_idx_ago[-1] - price_n_idx_ago[0]
        return float("nan")


@csp.graph
def momentum(
    x: ts[typing.Union[float, np.ndarray]], n: typing.Union[int, timedelta] = 2
) -> ts[typing.Union[float, np.ndarray]]:
    """
    :param x: time-series data
    :param n: look-back period
    """

    return _momentum(x, n)


@csp.node
def _calc_obv(x: ts[float], v: ts[float]) -> ts[float]:
    with csp.state():
        obv_prev = 0.0
        close_prev = float("nan")
    if csp.ticked(x) and csp.ticked(v):
        if math.isnan(close_prev):
            obv_new = 0
        else:
            y = 0
            if x > close_prev:
                y = v
            elif x < close_prev:
                y = -v
            obv_new = obv_prev + y
        obv_prev = obv_new
        close_prev = x
        return obv_new
        # obv = obv_prev + y


@csp.graph
def obv(x: ts[float], v: ts[float]) -> ts[float]:
    """
    :param x: close data
    :param v: volume data
    """
    obv = _calc_obv(x, v)
    return obv
