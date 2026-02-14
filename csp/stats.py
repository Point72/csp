from datetime import datetime, timedelta
from typing import Any, List, Optional, TypeVar, Union

import numpy as np

import csp
from csp import ts
from csp.lib import _cspnpstatsimpl, _cspstatsimpl
from csp.typing import Numpy1DArray, NumpyNDArray

__all__ = [
    "count",
    "unique",
    "first",
    "last",
    "sum",
    "prod",
    "mean",
    "gmean",
    "quantile",
    "median",
    "min",
    "max",
    "argmin",
    "argmax",
    "rank",
    "cov",
    "var",
    "corr",
    "stddev",
    "sem",
    "skew",
    "kurt",
    "ema",
    "ema_var",
    "ema_std",
    "ema_cov",
    "cov_matrix",
    "corr_matrix",
    "cross_sectional",
    "list_to_numpy",
    "numpy_to_list",
]


T = TypeVar("T")
U = TypeVar("U")


# Error messages
NP_SHAPE_ERROR = "Shape of the NumPy array was unknown at the time the trigger ticked."

"""
Base data processing nodes for statistical functions
"""


@csp.node(cppimpl=_cspstatsimpl._tick_window_updates)
def _tick_window_updates(
    x: ts[float], interval: int, trigger: ts[object], sampler: ts[object], reset: ts[object], recalc: ts[object]
) -> csp.Outputs(additions=ts[List[float]], removals=ts[List[float]]):
    raise NotImplementedError("_tick_window_updates only implemented in C++")
    return csp.output(additions=0, removals=0)


@csp.node(cppimpl=_cspstatsimpl._time_window_updates)
def _time_window_updates(
    x: ts[float], interval: timedelta, trigger: ts[object], sampler: ts[object], reset: ts[object], recalc: ts[object]
) -> csp.Outputs(additions=ts[List[float]], removals=ts[List[float]]):
    raise NotImplementedError("_time_window_updates only implemented in C++")
    return csp.output(additions=0, removals=0)


@csp.node(cppimpl=_cspnpstatsimpl._np_tick_window_updates)
def _np_tick_window_updates(
    x: ts[np.ndarray], interval: int, trigger: ts[object], sampler: ts[object], reset: ts[object], recalc: ts[object]
) -> csp.Outputs(additions=ts[List[np.ndarray]], removals=ts[List[np.ndarray]]):
    raise NotImplementedError("_np_tick_window_updates only implemented in C++")
    return csp.output(additions=0, removals=0)


@csp.node(cppimpl=_cspnpstatsimpl._np_time_window_updates)
def _np_time_window_updates(
    x: ts[np.ndarray],
    interval: timedelta,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    recalc: ts[object],
) -> csp.Outputs(additions=ts[List[np.ndarray]], removals=ts[List[np.ndarray]]):
    raise NotImplementedError("_np_time_window_updates only implemented in C++")
    return csp.output(additions=0, removals=0)


@csp.graph
def _window_updates(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int],
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    recalc: ts[object],
) -> csp.Outputs(additions=ts[List[Union[float, np.ndarray]]], removals=ts[List[Union[float, np.ndarray]]]):
    """
    :param x: the time-series data
    :param interval: a tick or timedelta interval to calculate over
    :param trigger: the computation trigger
    :param sampler: the series to use for nan checking
    :return: data additions and removals upon triggering, if they exist
    """

    # Allows for cppnodes to be different for time and tick
    if x.tstype.typ is float:
        if isinstance(interval, int):
            upd = _tick_window_updates(x, interval, trigger, sampler, reset, recalc)
        else:
            upd = _time_window_updates(x, interval, trigger, sampler, reset, recalc)
    elif x.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if isinstance(interval, int):
            upd = _np_tick_window_updates(x, interval, trigger, sampler, reset, recalc)
        else:
            upd = _np_time_window_updates(x, interval, trigger, sampler, reset, recalc)
    else:
        raise TypeError(
            f"Statistics methods are only valid for time-series of type float or np.ndarray[float]; given type {x.tstype.typ}"
        )

    additions = upd.additions
    removals = upd.removals

    return csp.output(additions=additions, removals=removals)


@csp.node(cppimpl=_cspstatsimpl._min_hit_by_tick)
def _min_hit_by_tick(x: ts["T"], min_window: int, trigger: ts[object]) -> ts[bool]:
    if csp.ticked(trigger):
        if csp.num_ticks(x) >= min_window:
            csp.make_passive(trigger)
            return True


@csp.graph
def _min_hit(x: ts["T"], min_window: Union[timedelta, int], trigger: ts[object]) -> ts[bool]:
    if isinstance(min_window, int):
        return _min_hit_by_tick(x, min_window, trigger)
    return csp.const(True, delay=min_window)


@csp.node(cppimpl=_cspstatsimpl._in_sequence_check)
def _in_sequence_check(x: ts["T"], y: ts["T"]):
    raise NotImplementedError("_in_sequence_check only implemented in C++")


@csp.node(cppimpl=_cspstatsimpl._discard_non_overlapping)
def _discard_non_overlapping(x: ts[float], y: ts[float]) -> csp.Outputs(x_sync=ts[float], y_sync=ts[float]):
    raise NotImplementedError("_discard_non_overlapping only implemented in C++")
    return csp.output(x_sync=0, y_sync=0)


@csp.node(cppimpl=_cspstatsimpl._sync_nan_f)
def _sync_nan_f(x: ts[float], y: ts[float]) -> csp.Outputs(x_sync=ts[float], y_sync=ts[float]):
    raise NotImplementedError("_sync_nan_f only implemented in C++")
    return csp.output(x_sync=0, y_sync=0)


@csp.node(cppimpl=_cspnpstatsimpl._sync_nan_np)
def _sync_nan_np(x: ts[np.ndarray], y: ts[np.ndarray]) -> csp.Outputs(x_sync=ts[np.ndarray], y_sync=ts[np.ndarray]):
    raise NotImplementedError("_sync_nan_np only implemented in C++")
    return csp.output(x_sync=0, y_sync=0)


@csp.graph
def _sync_nan(x: ts[Union[float, np.ndarray]], y: ts[Union[float, np.ndarray]]) -> csp.Outputs(
    x_sync=ts[Union[float, np.ndarray]], y_sync=ts[Union[float, np.ndarray]]
):
    return _sync_nan_f(x, y) if x.tstype.typ is float else _sync_nan_np(x, y)


@csp.node
def _combine_signal(x: ts["T"], y: ts["U"]) -> ts[bool]:
    if csp.ticked(x, y):
        return True


@csp.node
def _np_log(x: ts[np.ndarray]) -> ts[np.ndarray]:
    return np.log(x)


@csp.node
def _np_exp(x: ts[np.ndarray]) -> ts[np.ndarray]:
    return np.exp(x)


@csp.node(cppimpl=_cspnpstatsimpl._list_to_np)
def list_to_numpy(x: [ts[float]], fillna: bool = False) -> ts[Numpy1DArray[float]]:
    """
    x: listbasket of floats
    fillna: if True, unticked values will hold their previous value in the array.
        If False, unticked values are treated as NaN.
    """

    raise NotImplementedError("_list_to_np only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_to_list)
def numpy_to_list(x: ts[np.ndarray], n: int) -> csp.OutputBasket(List[ts[float]], shape="n"):
    raise NotImplementedError("_np_to_list only implemented in C++")
    return 0


"""
Function argument samplers
"""


def _setup(x, interval, min_window, trigger, sampler, reset, weights=None, ignore_weights=False, recalc=None):
    """
    Validates and sets function inputs for time-series stats
    """

    if interval == timedelta():
        raise ValueError(
            "Time specified interval needs to be positive. To specify an expanding window, use interval=None"
        )

    if interval is None:
        if recalc is not None:
            raise ValueError("The recalc parameter cannot be used with an expanding window (it is redundant)")
        interval = timedelta()

    if min_window is None:
        min_window = interval
    else:
        if type(min_window) is not type(interval):
            raise TypeError("Interval and min_window must be of the same type")

    if sampler is None:
        sampler = x

    if trigger is None:
        trigger = sampler

    if reset is not None:
        if recalc is not None:
            clear_stat = _combine_signal(reset, recalc)
        else:
            recalc = csp.null_ts(bool)
            clear_stat = reset
    else:
        reset = csp.null_ts(bool)
        if recalc is not None:
            clear_stat = recalc
        else:
            recalc = csp.null_ts(bool)
            clear_stat = csp.null_ts(bool)

    # Only filter on min_hit if necessary
    min_hit = None
    if (isinstance(min_window, int) and min_window > 1) or (
        isinstance(min_window, timedelta) and min_window > timedelta(0)
    ):
        min_hit = csp.default(_min_hit(sampler, min_window, trigger), False)

    series = x
    if x.tstype.typ is int:
        series = csp.cast_int_to_float(x)

    if weights is not None:
        if weights.tstype.typ is int:
            weights = csp.cast_int_to_float(weights)
        if not ignore_weights and weights.tstype.typ is not series.tstype.typ:
            raise ValueError(
                f"Weights and series must be of the same type: weights is {weights.tstype.typ}, series is {x.tstype.typ}"
            )

    updates = _window_updates(series, interval, trigger, sampler, reset, recalc)

    return series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stat


def _synchronize_bivariate(x, y, allow_non_overlapping):
    """
    If allow_non_overlapping=True, discard any out-of-sync ticks between and y. Else, raise an exception when this occurs.
    """
    if x is not y:
        if allow_non_overlapping:
            sync = _discard_non_overlapping(x, y)
            x, y = sync.x_sync, sync.y_sync
        else:
            _in_sequence_check(x, y)
    return x, y


def _bivariate_setup(
    x, y, interval, min_window, trigger, sampler, reset, weights=None, recalc=None, allow_non_overlapping=False
):
    """
    Sets up time-series window updates and triggers for a bivariate stats calculation
    """
    x, y = _synchronize_bivariate(x, y, allow_non_overlapping)
    x_upd = _setup(x, interval, min_window, trigger, sampler, reset, weights, False, recalc)[5]
    series, interval, min_window, trigger, min_hit, y_upd, sampler, reset, weights, recalc, clear_stat = _setup(
        y, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    return (
        series,
        interval,
        min_window,
        trigger,
        min_hit,
        x_upd,
        y_upd,
        sampler,
        reset,
        weights,
        recalc,
        clear_stat,
    )


def _validate_ema(alpha, span, com, halflife, adjust, horizon, recalc):
    """
    Validates and sets alpha for any ema-based function
    """
    if horizon and not adjust:
        raise ValueError("EMA with finite horizon must use adjusted EMA for weight normalization")

    param = [x for x in [alpha, span, com, halflife] if x is not None]
    if len(param) == 0:
        raise ValueError("One of alpha, span, com or halflife may be specified")
    elif len(param) > 1:
        raise ValueError("Only one of alpha, span, com or halflife can be specified")
    elif span:
        alpha = 2 / (span + 1)
    elif com:
        alpha = 1 / (1 + com)
    elif halflife:
        alpha = None

    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("Alpha parameter for EMA must be between 0 and 1")

    if not horizon:
        horizon = 0
        interval = None  # expanding
        min_window = timedelta(0)
        recalc = None
    else:
        interval = horizon
        min_window = 0

    return alpha, interval, min_window, horizon, recalc


"""
Utility nodes for the statistical API
"""


@csp.node(cppimpl=_cspstatsimpl._count)
def _count(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_count only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_count)
def _np_count(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_count only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._sum)
def _sum(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_sum only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._kahan_sum)
def _kahan_sum(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_kahan_sum only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_sum)
def _np_sum(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_sum only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_kahan_sum)
def _np_kahan_sum(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_kahan_sum only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._mean)
def _mean(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_mean only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_mean)
def _weighted_mean(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_mean only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_mean)
def _np_mean(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_mean only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._var)
def _var(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_var only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._sem)
def _sem(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_sem only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_var)
def _weighted_var(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    arg: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_var only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_sem)
def _weighted_sem(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    arg: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_sem only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_mean)
def _np_weighted_mean(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_mean only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._covar)
def _covar(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_covar only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._corr)
def _corr(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_covar)
def _weighted_covar(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    w_add: ts[List[float]],
    w_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_covar only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_corr)
def _weighted_corr(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    w_add: ts[List[float]],
    w_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_var)
def _np_var(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_var only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_sem)
def _np_sem(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_sem only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_covar)
def _np_covar(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_covar only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_corr)
def _np_corr(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_var)
def _np_weighted_var(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    arg: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_var only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_sem)
def _np_weighted_sem(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    arg: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_sem only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_covar)
def _np_weighted_covar(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    y_add: ts[List[np.ndarray]],
    y_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_covar only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_corr)
def _np_weighted_corr(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    y_add: ts[List[np.ndarray]],
    y_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_cov_matrix)
def _np_cov_matrix(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    ddof: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_cov only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_corr_matrix)
def _np_corr_matrix(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    ddof: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_cov_matrix)
def _np_weighted_cov_matrix(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[float]],
    w_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    ddof: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_cov only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_corr_matrix)
def _np_weighted_corr_matrix(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[float]],
    w_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    ddof: int,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_corr only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._skew)
def _skew(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    # arg is bias
    raise NotImplementedError("_skew only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._kurt)
def _kurt(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg1: bool,
    arg2: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    # arg1 is bias, arg2 is excess
    raise NotImplementedError("_kurt only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_skew)
def _weighted_skew(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_skew only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._weighted_kurt)
def _weighted_kurt(
    x_add: ts[List[float]],
    x_rem: ts[List[float]],
    y_add: ts[List[float]],
    y_rem: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    arg1: bool,
    arg2: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_weighted_kurt only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_skew)
def _np_skew(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    # arg is bias
    raise NotImplementedError("_np_skew only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_kurt)
def _np_kurt(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg1: bool,
    arg2: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    # arg1 is bias, arg2 is excess
    raise NotImplementedError("_np_kurt only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_skew)
def _np_weighted_skew(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_skew only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_weighted_kurt)
def _np_weighted_kurt(
    x_add: ts[List[np.ndarray]],
    x_rem: ts[List[np.ndarray]],
    w_add: ts[List[np.ndarray]],
    w_rem: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    arg1: bool,
    arg2: bool,
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_weighted_kurt only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._first)
def _first(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_first node only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_first)
def _np_first(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_first node only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._last)
def _last(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_last node only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_last)
def _np_last(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_last only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._unique)
def _unique(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg: int,
) -> ts[float]:
    raise NotImplementedError("_unique only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_unique)
def _np_unique(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_unique only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._prod)
def _prod(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[float]:
    raise NotImplementedError("_prod only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_prod)
def _np_prod(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_prod only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._quantile)
def _quantile(
    additions: ts[List[float]],
    removals: ts[List[float]],
    quants: List[float],
    nq: int,
    interpolation_type: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> csp.OutputBasket(List[ts[float]], shape="nq"):
    raise NotImplementedError("_quantile only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_quantile)
def _np_quantile(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    quants: List[float],
    nq: int,
    interpolation_type: int,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> csp.OutputBasket(List[ts[np.ndarray]], shape="nq"):
    raise NotImplementedError("_np_quantile only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._min_max)
def _min_max(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg: bool,
) -> ts[float]:
    raise NotImplementedError("_min_max only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_min_max)
def _np_min_max(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_min_max only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._rank)
def _rank(
    additions: ts[List[float]],
    removals: ts[List[float]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg1: int,
    arg2: int,
) -> ts[float]:
    raise NotImplementedError("_rank only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_rank)
def _np_rank(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
    ignore_na: bool,
    arg1: int,
    arg2: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_rank only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._arg_min_max)
def _arg_min_max(
    x: ts[float],
    removals: ts[List[float]],
    max: bool,
    recent: bool,
    trigger: ts[object],
    reset: ts[object],
    sampler: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[datetime]:
    raise NotImplementedError("_arg_min_max only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_arg_min_max)
def _np_arg_min_max(
    x: ts[np.ndarray],
    removals: ts[List[np.ndarray]],
    max: bool,
    recent: bool,
    trigger: ts[object],
    reset: ts[object],
    sampler: ts[object],
    min_data_points: int,
    ignore_na: bool,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_arg_min_max only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_compute)
def _ema_compute(
    additions: ts[List[float]],
    removals: ts[List[float]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_compute only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_compute)
def _np_ema_compute(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_compute only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_adjusted)
def _ema_adjusted(
    additions: ts[List[float]],
    removals: ts[List[float]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_adjusted only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_adjusted)
def _np_ema_adjusted(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_adjusted only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_halflife)
def _ema_halflife(
    x: ts[float],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_halflife only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_halflife_adjusted)
def _ema_halflife_adjusted(
    x: ts[float],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_halflife_adjusted only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_halflife)
def _np_ema_halflife(
    x: ts[np.ndarray],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_halflife only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_halflife_adjusted)
def _np_ema_halflife_adjusted(
    x: ts[np.ndarray],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_halflife_adjusted only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_halflife_debias)
def _ema_halflife_debias(
    x: ts[float],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_halflife_debias only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_halflife_debias)
def _np_ema_halflife_debias(
    x: ts[np.ndarray],
    halflife: timedelta,
    adjust: bool,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_halflife_debias only implemented in C++")
    return 0


@csp.node(cppimpl=_cspstatsimpl._ema_alpha_debias)
def _ema_alpha_debias(
    additions: ts[List[float]],
    removals: ts[List[float]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[float]:
    raise NotImplementedError("_ema_alpha_debias only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_ema_alpha_debias)
def _np_ema_alpha_debias(
    additions: ts[List[np.ndarray]],
    removals: ts[List[np.ndarray]],
    alpha: float,
    ignore_na: bool,
    horizon: int,
    adjust: bool,
    trigger: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_ema_alpha_debias only implemented in C++")
    return 0


@csp.graph
def _ema_debias(
    x: ts[Union[float, np.ndarray]],
    additions: ts[Union[List[float], List[np.ndarray]]],
    removals: ts[Union[List[float], List[np.ndarray]]],
    alpha: float,
    ignore_na: bool,
    adjust: bool,
    halflife: timedelta,
    horizon: int,
    trigger: ts[object],
    sampler: ts[object],
    reset: ts[object],
    min_data_points: int,
) -> ts[Union[float, np.ndarray]]:
    if alpha:
        if not horizon:
            horizon = 0
        if x.tstype.typ is float:
            return _ema_alpha_debias(
                additions, removals, alpha, ignore_na, horizon, adjust, trigger, reset, min_data_points
            )
        else:
            return _np_ema_alpha_debias(
                additions, removals, alpha, ignore_na, horizon, adjust, trigger, reset, min_data_points
            )

    if x.tstype.typ is float:
        return _ema_halflife_debias(x, halflife, adjust, trigger, sampler, reset, min_data_points)

    return _np_ema_halflife_debias(x, halflife, adjust, trigger, sampler, reset, min_data_points)


@csp.node(cppimpl=_cspstatsimpl._cross_sectional_as_list)
def _cross_sectional_as_list(
    additions: ts[List[float]], removals: ts[List[float]], trigger: ts[object], reset: ts[object]
) -> ts[List[float]]:
    raise NotImplementedError("_cross_sectional_as_list only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._cross_sectional_as_np)
def _cross_sectional_as_np(
    additions: ts[List[float]], removals: ts[List[float]], trigger: ts[object], reset: ts[object]
) -> ts[np.ndarray]:
    raise NotImplementedError("_cross_sectional_as_np only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_cross_sectional_as_list)
def _np_cross_sectional_as_list(
    additions: ts[List[np.ndarray]], removals: ts[List[np.ndarray]], trigger: ts[object], reset: ts[object]
) -> ts[List[np.ndarray]]:
    raise NotImplementedError("_np_cross_sectional_as_list only implemented in C++")
    return 0


@csp.node(cppimpl=_cspnpstatsimpl._np_cross_sectional_as_np)
def _np_cross_sectional_as_np(
    additions: ts[List[np.ndarray]], removals: ts[List[np.ndarray]], trigger: ts[object], reset: ts[object]
) -> ts[np.ndarray]:
    raise NotImplementedError("_np_cross_sectional_as_np only implemented in C++")
    return 0


"""
Execution functions for code modularity
"""


@csp.graph
def _execute_stats(edge: Any = None, min_hit: ts[bool] = None) -> ts[Union[float, datetime, np.ndarray]]:
    # only filter on min_hit if we need to
    if min_hit is not None:
        edge = csp.filter(min_hit, edge)
    return edge


@csp.graph
def _arg_minmax(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    return_most_recent: bool = True,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    max: bool = True,
) -> ts[Union[datetime, np.ndarray]]:
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    edge = None
    if series.tstype.typ is float:
        edge = _arg_min_max(
            series, updates.removals, max, return_most_recent, trigger, reset, sampler, min_data_points, ignore_na
        )
    else:
        edge = _np_arg_min_max(
            series, updates.removals, max, return_most_recent, trigger, reset, sampler, min_data_points, ignore_na
        )

    return _execute_stats(edge, min_hit)


"""
Basic Statistics API
"""


@csp.graph
def count(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the count of (non-nan) ticks in the window, either including/ignoring nan values.

    Inputs
    x:          the time series data, either of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  flag to control NaN handling. If False, NaN ticks will poison the count, returning NaN.
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    edge = None
    if series.tstype.typ is float:
        edge = _count(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)
    else:
        edge = _np_count(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)

    if min_hit is not None:
        return csp.filter(min_hit, edge)
    return edge


@csp.graph
def unique(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    precision: int = 10,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the number of unique non-nan values in the current window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    precision:  the number of decimal places at which two floats are considered equal

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    if series.tstype.typ is float:
        edge = _unique(updates.additions, updates.removals, trigger, reset, min_data_points, True, precision)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_unique(updates.additions, updates.removals, trigger, reset, min_data_points, True, precision)

    if min_hit is not None:
        return csp.filter(min_hit, edge)
    return edge


@csp.graph
def first(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    ignore_na: bool = True,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the first non-nan value currently within the window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    ignore_na:  if True, will return the first non-nan value in the window. If False, will return the first value in the window
    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    if series.tstype.typ is float:
        edge = _first(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_first(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)

    if min_hit is not None:
        return csp.filter(min_hit, edge)
    return edge


@csp.graph
def last(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the last value currently within the window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will return the last non-nan value in the window. If False, will return the last value in the window
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    if series.tstype.typ is float:
        edge = _last(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_last(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na)

    if min_hit is not None:
        return csp.filter(min_hit, edge)
    return edge


@csp.graph
def sum(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    precise: bool = False,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the sum of values over a given window.

    Inputs
    x:          the time series data, of either type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    precise:    if True, the Kahan summation is used for added numerical stability (although less efficient)
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    weights:    another time-series which specifies the weights to use on each x value, if a weighted sum is desired
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    recalc:     another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    edge = None
    data = x
    if weights is not None:
        data = x * weights

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stat = _setup(
        data, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    if series.tstype.typ is float:
        if precise:
            edge = _kahan_sum(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)
        else:
            edge = _sum(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)

    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if precise:
            edge = _np_kahan_sum(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)
        edge = _np_sum(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def mean(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the mean over a rolling window.

    Inputs
    x:          the time series data, of either type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    weights:    another time-series which specifies the weights to use on each x value, if a weighted mean is desired
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    recalc:     another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )
    edge = None

    if series.tstype.typ is float:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_mean(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                trigger,
                clear_stat,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _mean(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)

    elif x.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_mean(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                trigger,
                clear_stat,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_mean(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def prod(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the product over a rolling window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    recalc:     another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, recalc=recalc
    )

    if series.tstype.typ is float:
        edge = _prod(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_prod(updates.additions, updates.removals, trigger, clear_stat, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


# Not a graph since it has two different return types: list-basket and time-series
def quantile(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    quant: Union[float, List[float]] = None,
    min_window: Union[timedelta, int] = None,
    interpolate: str = "linear",
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
):
    """

    Returns the specified quantile in the given window.
    If provided a single time-series with a single quantile, it will return the quantile as a time-series.
    If provided a single time-series with a list of quantiles, it will return the quantiles as a listbasket.
    If provided a NumPy array time series, it will return the quantiles as a single NumPy array or listbasket of NumPy arrays.

    Inputs
    x:              the time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    quant:          the quantile or list of quantiles to compute. If given a list, the output will be a listbasket containing each desired quantile.
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    interpolate:    the interpolation method to use. One of: linear, lower, higher, midpoint, nearest
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    interpolate_methods = {"linear": 0, "lower": 1, "higher": 2, "midpoint": 3, "nearest": 4}
    if interpolate not in interpolate_methods:
        raise ValueError("The interpolation method provided is not valid. Please consult the documentation")

    # Ensure quantiles are a list and accessible as a constant time-series
    if quant is None:
        raise ValueError("At least one quantile value must be provided")
    if isinstance(quant, float):
        quant = [quant]
    nq = len(quant)

    interpolation_type = interpolate_methods[interpolate]

    if series.tstype.typ is float:
        if nq == 1:
            edge = _quantile(
                updates.additions,
                updates.removals,
                quant,
                nq,
                interpolation_type,
                trigger,
                reset,
                min_data_points,
                ignore_na,
            )[0]
            return _execute_stats(edge, min_hit)
        else:
            edge = _quantile(
                updates.additions,
                updates.removals,
                quant,
                nq,
                interpolation_type,
                trigger,
                reset,
                min_data_points,
                ignore_na,
            )
            return [_execute_stats(edge[i], min_hit) for i in range(nq)]
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if nq == 1:
            edge = _np_quantile(
                updates.additions,
                updates.removals,
                quant,
                nq,
                interpolation_type,
                trigger,
                reset,
                min_data_points,
                ignore_na,
            )[0]
            return _execute_stats(edge, min_hit)
        else:
            edge = _np_quantile(
                updates.additions,
                updates.removals,
                quant,
                nq,
                interpolation_type,
                trigger,
                reset,
                min_data_points,
                ignore_na,
            )
            return [_execute_stats(edge[i], min_hit) for i in range(nq)]


def min_max(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    max: bool = True,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
):
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    if series.tstype.typ is float:
        edge = _min_max(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na, max)
        return _execute_stats(edge, min_hit)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_min_max(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na, max)
        return _execute_stats(edge, min_hit)


@csp.graph
def max(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the maximum value within a given window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return min_max(**locals(), max=True)


@csp.graph
def min(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the minimum value within a given window.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return min_max(**locals(), max=False)


@csp.graph
def rank(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    method: str = "min",
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    na_option: str = "keep",
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the rank (0-indexed) of the last tick in relation to all other values in the interval.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    method:     the method to use to rank groups of records that have the same value
                "min": the lowest rank in the group is returned i.e. if the window data is [1,2,2,3] and the last tick is 2, then rank=1
                "max": the highest rank in the group is returned i.e. if   the window data is [1,2,2,3] and the last tick is 2, then rank=3
                "avg": the average rank in the group is returned i.e. if   the window data is [1,2,2,3] and the last tick is 2, then rank=2
                    Note: the avg method must run both min and max, and thus is slower than the other two options
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    na_option:  how to rank a NaN value
                "keep": when a NaN value is encountered, return a NaN rank
                "last": when a NaN value is encountered, return the rank of the last valid non-NaN entry
    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    rank_methods = {"min": 0, "max": 1, "avg": 2}
    if method not in rank_methods:
        raise ValueError("The rank method must be one of: min, max, avg. Please consult the documentation")
    rmethod = rank_methods[method]

    na_options = {"keep": 0, "last": 1}
    if na_option not in na_options:
        raise ValueError("The NaN option must be one of: keep, last. Please consult the documentation")
    na_opt = na_options[na_option]

    if series.tstype.typ is float:
        edge = _rank(updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na, rmethod, na_opt)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        edge = _np_rank(
            updates.additions, updates.removals, trigger, reset, min_data_points, ignore_na, rmethod, na_opt
        )

    return _execute_stats(edge, min_hit)


@csp.graph
def argmax(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    return_most_recent: bool = True,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[datetime, np.ndarray]]:
    """

    Returns the datetime at which the maximum value in the interval ticked.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    return_most_recent: if True, in the case of a tie, the most recent time will be returned. Else, the least recent time will be returned.
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return _arg_minmax(**locals(), max=True)


@csp.graph
def argmin(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    return_most_recent: bool = True,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[datetime, np.ndarray]]:
    """

    Returns the datetime at which the minimum value in the interval ticked.

    Inputs
    x:          the time series data, of type float or np.ndarray
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    return_most_recent: if True, in the case of a tie, the most recent time will be returned. Else, the least recent time will be returned.
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return _arg_minmax(**locals(), max=False)


@csp.graph
def gmean(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the geometric mean of a strictly positive time series over a rolling window.

    Inputs
    x:          the time series data of type float or np.ndarray, and strictly positive
    interval:   the window interval (either time or tick specified)
    min_window: the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:  if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:    another time-series which specifies when you want to recalculate the statistic
    sampler:    another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                the data point is treated as NaN
    reset:      another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    if x.tstype.typ in [float, int]:
        return csp.exp(
            mean(
                csp.ln(x),
                interval,
                min_window,
                ignore_na,
                trigger,
                sampler=sampler,
                reset=reset,
                min_data_points=min_data_points,
            )
        )
    elif x.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        return _np_exp(
            mean(
                _np_log(x),
                interval,
                min_window,
                ignore_na,
                trigger,
                sampler=sampler,
                reset=reset,
                min_data_points=min_data_points,
            )
        )


@csp.graph
def median(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the median value in the given window.

    Inputs
    x:              the time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return quantile(
        x,
        interval,
        0.5,
        min_window,
        "midpoint",
        ignore_na,
        trigger,
        sampler,
        reset=reset,
        min_data_points=min_data_points,
    )


"""
Moment-Based Statistics
"""


@csp.graph
def cov(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
    allow_non_overlapping: bool = False,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the covariance between two in-sequence time-series within the given window. If the time-series are of type np.ndarray, the covariance is calculated elementwise.

    Inputs
    x:                      time series data, of type float or np.ndarray
    y:                      time series data, of type float or np.ndarray, which ticks at the same time as x
    interval:               the window interval (either time or tick specified)
    min_window:             the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ddof:                   delta degrees of freedom
    ignore_na:              if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:                another time-series which specifies when you want to recalculate the statistic
    weights:                another time-series which specifies the weights to use on each x value, if a weighted covariance is desired
    sampler:                another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                            the data point is treated as NaN
    reset:                  another time-series which will clear the data in the window when it ticks
    recalc:                 another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points:        minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    allow_non_overlapping:  if True, discard any ticks of x and y that occur out-of-sync with one another. If False, raise an exception on any out-of-sync ticks.

    """

    (
        series,
        interval,
        min_window,
        trigger,
        min_hit,
        x_upd,
        y_upd,
        sampler,
        reset,
        weights,
        recalc,
        clear_stat,
    ) = _bivariate_setup(x, y, interval, min_window, trigger, sampler, reset, weights, recalc, allow_non_overlapping)

    # Use same "debiasing" for weighted/non-weighted
    edge = None
    if series.tstype.typ is float:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_covar(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                ddof,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _covar(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                trigger,
                clear_stat,
                ddof,
                min_data_points,
                ignore_na,
            )
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_covar(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                ddof,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_covar(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                trigger,
                clear_stat,
                ddof,
                min_data_points,
                ignore_na,
            )

    return _execute_stats(edge, min_hit)  # no need to filter


@csp.graph
def cov_matrix(
    x: ts[np.ndarray],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[float] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[np.ndarray]:
    """

    Returns the covariance matrix of an array of random variables within the given window.

    Inputs
    x:              time series data, of type Numpy1DArray representing an array of random variables
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ddof:           delta degrees of freedom
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted covariance is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    # Covariance matrix expects column vector inputs
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, True, recalc
    )
    if weights is not None:
        w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
        edge = _np_weighted_cov_matrix(
            updates.additions,
            updates.removals,
            w_upd.additions,
            w_upd.removals,
            trigger,
            clear_stat,
            ddof,
            min_data_points,
            ignore_na,
        )
    else:
        edge = _np_cov_matrix(
            updates.additions, updates.removals, trigger, clear_stat, ddof, min_data_points, ignore_na
        )

    return _execute_stats(edge, min_hit)


@csp.graph
def var(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the variance within the given window.

    Inputs
    x:              time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ddof:           delta degrees of freedom
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted variance is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    edge = None
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stats = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    if series.tstype.typ is float:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_var(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                ddof,
                trigger,
                clear_stats,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _var(updates.additions, updates.removals, trigger, clear_stats, ddof, min_data_points, ignore_na)

    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        # NumPy array element-wise variance
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_var(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                ddof,
                trigger,
                clear_stats,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_var(updates.additions, updates.removals, trigger, clear_stats, ddof, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def stddev(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the standard deviation within the given window.

    Inputs
    x:              time series data, of type float, or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ddof:           delta degrees of freedom
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted standard deviation is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return var(**locals()) ** (1 / 2)


@csp.graph
def sem(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the standard error of the mean within the given window.

    Inputs
    x:              time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ddof:           delta degrees of freedom
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted SEM is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    edge = None
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stats = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    if series.tstype.typ is float:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_sem(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                ddof,
                trigger,
                clear_stats,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _sem(updates.additions, updates.removals, trigger, clear_stats, ddof, min_data_points, ignore_na)

    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_sem(
                updates.additions,
                updates.removals,
                weight_updates.additions,
                weight_updates.removals,
                ddof,
                trigger,
                clear_stats,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_sem(updates.additions, updates.removals, trigger, clear_stats, ddof, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def corr(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
    allow_non_overlapping: bool = False,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the correlation between x and y within the given window. If the time-series are of type np.ndarray, the correlation is calculated elementwise.

    Inputs
    x:                      time series data, of type float or np.ndarray
    y:                      time series data, of type float or np.ndarray, which ticks at the same time x ticks
    interval:               the window interval (either time or tick specified)
    min_window:             the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:              if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:                another time-series which specifies when you want to recalculate the statistic
    weights:                another time-series which specifies the weights to use on each x value, if a weighted correlation is desired
    sampler:                another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                            the data point is treated as NaN
    reset:                  another time-series which will clear the data in the window when it ticks
    recalc:                 another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points:        minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    allow_non_overlapping:  if True, discard any ticks of x and y that occur out-of-sync with one another. If False, raise an exception on any out-of-sync ticks.

    """
    (
        series,
        interval,
        min_window,
        trigger,
        min_hit,
        x_upd,
        y_upd,
        sampler,
        reset,
        weights,
        recalc,
        clear_stat,
    ) = _bivariate_setup(x, y, interval, min_window, trigger, sampler, reset, weights, recalc, allow_non_overlapping)

    if series.tstype.typ is float:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_corr(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                weight_updates.additions,
                weight_updates.removals,
                trigger,
                clear_stat,
                0,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _corr(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                trigger,
                clear_stat,
                min_data_points,
                ignore_na,
            )
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_corr(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                weight_updates.additions,
                weight_updates.removals,
                trigger,
                clear_stat,
                0,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_corr(
                x_upd.additions,
                x_upd.removals,
                y_upd.additions,
                y_upd.removals,
                trigger,
                clear_stat,
                min_data_points,
                ignore_na,
            )

    return _execute_stats(edge, min_hit)


@csp.graph
def corr_matrix(
    x: ts[np.ndarray],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[float] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[np.ndarray]:
    """

    Returns the correlation matrix of an array of random variableswithin the given window.

    Inputs
    x:              time series data, of type Numpy1DArray representing an array of random variables
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted correlation matrix is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """
    series, interval, min_window, trigger, min_hit, updates, sampler, reset, weights, recalc, clear_stats = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, True, recalc
    )

    if weights is not None:
        weight_updates = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
        edge = _np_weighted_corr_matrix(
            updates.additions,
            updates.removals,
            weight_updates.additions,
            weight_updates.removals,
            trigger,
            clear_stats,
            0,
            min_data_points,
            ignore_na,
        )
    else:
        edge = _np_corr_matrix(updates.additions, updates.removals, trigger, clear_stats, 0, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def skew(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    bias: bool = False,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the skew within the given window.

    Inputs
    x:              time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    bias:           if True, calculates biased skew using the definition. If False, uses the adjusted Fisher-Pearson correction so that the skew is Gaussian-unbiased.
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted skew is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, x_upd, sampler, reset, weights, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    # Use same "debiasing" for weighted/non-weighted
    edge = None
    if series.tstype.typ is float:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_skew(
                x_upd.additions,
                x_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                bias,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _skew(x_upd.additions, x_upd.removals, trigger, clear_stat, bias, min_data_points, ignore_na)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_skew(
                x_upd.additions,
                x_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                bias,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_skew(x_upd.additions, x_upd.removals, trigger, clear_stat, bias, min_data_points, ignore_na)

    return _execute_stats(edge, min_hit)


@csp.graph
def kurt(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    excess: bool = True,
    bias: bool = False,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the kurtosis within the given window.

    Inputs
    x:              time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    ignore_na:      if True, will treat NaN values as missing data. If False, a NaN present in the window will make the computed statistic NaN as well
    excess:         if True, computes excess kurtosis. If False, computes standard kurtosis
    bias:           if True, calculates biased kurtosis using the definition. If False, uses the adjusted Fisher-Pearson correction so that the kurtosis is Gaussian-unbiased.
    trigger:        another time-series which specifies when you want to recalculate the statistic
    weights:        another time-series which specifies the weights to use on each x value, if a weighted kurtosis is desired
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    series, interval, min_window, trigger, min_hit, x_upd, sampler, reset, weights, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, weights, False, recalc
    )

    edge = None
    if series.tstype.typ is float:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _weighted_kurt(
                x_upd.additions,
                x_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                bias,
                excess,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _kurt(x_upd.additions, x_upd.removals, trigger, clear_stat, bias, excess, min_data_points, ignore_na)
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if weights is not None:
            w_upd = _window_updates(csp.sample(sampler, weights), interval, trigger, sampler, reset, recalc)
            edge = _np_weighted_kurt(
                x_upd.additions,
                x_upd.removals,
                w_upd.additions,
                w_upd.removals,
                trigger,
                clear_stat,
                bias,
                excess,
                min_data_points,
                ignore_na,
            )
        else:
            edge = _np_kurt(
                x_upd.additions, x_upd.removals, trigger, clear_stat, bias, excess, min_data_points, ignore_na
            )

    return _execute_stats(edge, min_hit)


"""
EMA Statistics
"""


@csp.graph
def ema(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: timedelta = None,
    adjust: bool = True,
    horizon: int = None,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the exponential moving avergae of a time series.

    Inputs
    x:              time series data, of type float or np.ndarray
    min_periods:    the minimum number of data points before statistics are returned
    alpha:          specify the decay parameter in terms of alpha
    span:           specify the decay parameter in terms of span
    com:            specify the decay parameter in terms of com
    halflife:       specify the decay parameter in terms of halflife
    adjust:         if True, an adjusted EMA will be computed. If False, a standard (unadjusted) EMA will be computed
    horizon:        if specified, values that are older than the horizon will be removed entirely from the computation (essentially making EMA a window computation)
    ignore_na:      if True, NaNs will be ignored and have no effect on the computation. If False, a NaN will shift the observation window once new non-NaN data comes in
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         only valid when a finite-horizon EMA is used. Another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    alpha, interval, min_window, horizon, recalc = _validate_ema(alpha, span, com, halflife, adjust, horizon, recalc)
    series, _, _, trigger, _, updates, sampler, reset, _, recalc, clear_stat = _setup(
        x, interval, min_window, trigger, sampler, reset, None, False, recalc
    )

    edge = None
    if series.tstype.typ is float:
        if halflife:
            # ignore na does not matter for the halflife case; adjust parameter does not need to be passed here either, set as False
            if adjust:
                edge = _ema_halflife_adjusted(series, halflife, False, trigger, sampler, reset, min_data_points)
            else:
                edge = _ema_halflife(series, halflife, False, trigger, sampler, reset, min_data_points)
        elif adjust:
            edge = _ema_adjusted(
                updates.additions,
                updates.removals,
                alpha,
                ignore_na,
                horizon,
                adjust,
                trigger,
                clear_stat,
                min_data_points,
            )
        else:
            edge = _ema_compute(
                updates.additions, updates.removals, alpha, ignore_na, horizon, adjust, trigger, reset, min_data_points
            )
    elif series.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        if halflife:
            if adjust:
                edge = _np_ema_halflife_adjusted(series, halflife, False, trigger, sampler, reset, min_data_points)
            else:
                edge = _np_ema_halflife(series, halflife, False, trigger, sampler, reset, min_data_points)
        elif adjust:
            edge = _np_ema_adjusted(
                updates.additions,
                updates.removals,
                alpha,
                ignore_na,
                horizon,
                adjust,
                trigger,
                clear_stat,
                min_data_points,
            )
        else:
            edge = _np_ema_compute(
                updates.additions, updates.removals, alpha, ignore_na, horizon, adjust, trigger, reset, min_data_points
            )

    # Avoid unnecessary min_hit wiring if possible
    if min_periods > 1:
        return csp.filter(_min_hit_by_tick(series, min_periods, trigger), edge)
    else:
        return edge


@csp.graph
def ema_cov(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: timedelta = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
    allow_non_overlapping: bool = False,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the exponential moving covariance between two time series.

    Inputs
    x:                      time series data, of type float or np.ndarray
    y:                      time series data, of type float or np.ndarray, which ticks at the same time as x
    min_periods:            the minimum number of data points before statistics are returned
    alpha:                  specify the decay parameter in terms of alpha
    span:                   specify the decay parameter in terms of span
    com:                    specify the decay parameter in terms of com
    halflife:               specify the decay parameter in terms of halflife
    adjust:                 if True, an adjusted EMA will be computed. If False, a standard (unadjusted) EMA will be computed
    horizon:                if specified, values that are older than the horizon will be removed entirely from the computation (essentially making EMA a window computation)
    bias:                   if True, a biased EMA covariance is computed. If False, the covariance estimate is unbiased
    ignore_na:              if True, NaNs will be ignored and have no effect on the computation. If False, a NaN will shift the observation window once new non-NaN data comes in
    trigger:                another time-series which specifies when you want to recalculate the statistic
    sampler:                another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                            the data point is treated as NaN
    reset:                  another time-series which will clear the data in the window when it ticks
    recalc:                 only valid when a finite-horizon EMA is used. Another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points:        minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.
    allow_non_overlapping:  if True, discard any ticks of x and y that occur out-of-sync with one another. If False, raise an exception on any out-of-sync ticks.

    """

    alpha, interval, min_window, debias_horizon, recalc = _validate_ema(
        alpha, span, com, halflife, adjust, horizon, recalc
    )

    x, y = _synchronize_bivariate(x, y, allow_non_overlapping)
    if x is not y:
        sync = _sync_nan(x, y)
        x, y = sync.x_sync, sync.y_sync

    bias_cov = ema(
        x * y,
        min_periods,
        alpha,
        halflife=halflife,
        adjust=adjust,
        horizon=horizon,
        ignore_na=ignore_na,
        trigger=trigger,
        sampler=sampler,
        reset=reset,
        recalc=recalc,
        min_data_points=min_data_points,
    ) - ema(
        x,
        min_periods,
        alpha,
        halflife=halflife,
        adjust=adjust,
        horizon=horizon,
        ignore_na=ignore_na,
        trigger=trigger,
        sampler=sampler,
        reset=reset,
        recalc=recalc,
        min_data_points=min_data_points,
    ) * ema(
        y,
        min_periods,
        alpha,
        halflife=halflife,
        adjust=adjust,
        horizon=horizon,
        ignore_na=ignore_na,
        trigger=trigger,
        sampler=sampler,
        reset=reset,
        recalc=recalc,
        min_data_points=min_data_points,
    )
    if bias:
        return bias_cov
    else:
        series, _, _, trigger, min_hit, updates, sampler, reset, _, recalc, clear_stat = _setup(
            x, interval, min_window, trigger, sampler, reset, None, False, recalc
        )
        factor = _ema_debias(
            series,
            updates.additions,
            updates.removals,
            alpha,
            ignore_na,
            adjust,
            halflife,
            debias_horizon,
            trigger,
            sampler,
            clear_stat,
            min_data_points,
        )
        return csp.multiply(factor, bias_cov)


@csp.graph
def ema_var(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: timedelta = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the exponential moving variance of a time series.

    Inputs
    x:              time series data, of type float or np.ndarray
    min_periods:    the minimum number of data points before statistics are returned
    alpha:          specify the decay parameter in terms of alpha
    span:           specify the decay parameter in terms of span
    com:            specify the decay parameter in terms of com
    halflife:       specify the decay parameter in terms of halflife
    adjust:         if True, an adjusted EMA will be computed. If False, a standard (unadjusted) EMA will be computed
    horizon:        if specified, values that are older than the horizon will be removed entirely from the computation (essentially making EMA a window computation)
    bias:           if True, a biased EMA variance is computed. If False, the variance estimate is unbiased
    ignore_na:      if True, NaNs will be ignored and have no effect on the computation. If False, a NaN will shift the observation window once new non-NaN data comes in
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         only valid when a finite-horizon EMA is used. Another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    edge = None
    if x.tstype.typ in [int, float]:
        edge = csp.max(
            ema_cov(
                x,
                x,
                min_periods,
                alpha,
                span,
                com,
                halflife,
                adjust,
                horizon,
                bias,
                ignore_na,
                trigger,
                sampler,
                reset,
                recalc,
                min_data_points,
            ),
            csp.const(0.0),
        )
    elif x.tstype.typ in [Numpy1DArray[float], NumpyNDArray[float]]:
        nn_array = lambda x: np.clip(x, a_min=0, a_max=None)  # noqa: E731
        edge = csp.apply(
            ema_cov(
                x,
                x,
                min_periods,
                alpha,
                span,
                com,
                halflife,
                adjust,
                horizon,
                bias,
                ignore_na,
                trigger,
                sampler,
                reset,
                recalc,
                min_data_points,
            ),
            nn_array,
            NumpyNDArray[float],
        )

    return edge


@csp.graph
def ema_std(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: timedelta = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) -> ts[Union[float, np.ndarray]]:
    """

    Returns the exponential moving standard deviation of a time series.

    Inputs
    x:              time series data, of type float or np.ndarray
    min_periods:    the minimum number of data points before statistics are returned
    alpha:          specify the decay parameter in terms of alpha
    span:           specify the decay parameter in terms of span
    com:            specify the decay parameter in terms of com
    halflife:       specify the decay parameter in terms of halflife
    adjust:         if True, an adjusted EMA will be computed. If False, a standard (unadjusted) EMA will be computed
    horizon:        if specified, values that are older than the horizon will be removed entirely from the computation (essentially making EMA a window computation)
    bias:           if True, standard deviation is based on a biased EMA variance estimate. If False, the variance estimate is unbiased
    ignore_na:      if True, NaNs will be ignored and have no effect on the computation. If False, a NaN will shift the observation window once new non-NaN data comes in
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks
    recalc:         only valid when a finite-horizon EMA is used. Another time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
    min_data_points: minimum number of current ticks in the interval needed for a valid computation. If there are fewer ticks, NaN is returned.

    """

    return ema_var(**locals()) ** (1 / 2)


@csp.graph
def cross_sectional(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[int, timedelta] = None,
    as_numpy: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
) -> ts[Union[np.ndarray, List[float], List[np.ndarray]]]:
    """

    Returns all data present in the current window so that users can apply their own cross-sectional calculations.

    Inputs
    x:              time series data, of type float or np.ndarray
    interval:       the window interval (either time or tick specified)
    min_window:     the minimum window (either time or tick specified) before statistics are returned. Must be the same type as interval
    as_numpy:       if True, the data will be returned as a NumPy array instead of a list.
                        -- For a single-valued time series, this is a one-dimensional NumPy array
                        -- For a NumPy array time series, this is a NumPy array of one extra dimension
    trigger:        another time-series which specifies when you want to recalculate the statistic
    sampler:        another time-series which specifies when x should tick. If x ticks when sampler does not, the data is ignored. If sampler ticks when x does not,
                    the data point is treated as NaN
    reset:          another time-series which will clear the data in the window when it ticks

    """

    series, interval, min_window, trigger, min_hit, updates, sampler, reset, _, _, _ = _setup(
        x, interval, min_window, trigger, sampler, reset
    )

    if series.tstype.typ is float:
        if as_numpy:
            edge = _cross_sectional_as_np(updates.additions, updates.removals, trigger, reset)
        else:
            edge = _cross_sectional_as_list(updates.additions, updates.removals, trigger, reset)
    else:
        if as_numpy:
            edge = _np_cross_sectional_as_np(updates.additions, updates.removals, trigger, reset)
        else:
            edge = _np_cross_sectional_as_list(updates.additions, updates.removals, trigger, reset)

    if min_hit is not None:
        return csp.filter(min_hit, edge)

    return edge
