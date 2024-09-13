from datetime import timedelta

import pandas

import csp
from csp import ts


@csp.node
def _pandas_at(
    trigger: ts[object], xs: {object: ts[object]}, window: object, has_tindex: bool, know_all_valid: bool
) -> ts[pandas.DataFrame]:
    """Low level implementation for creating a pandas frame out of the buffers of several time series."""
    with csp.start():
        csp.make_passive(xs)
        if isinstance(window, int):
            assert window > 0
            csp.set_buffering_policy(xs, tick_count=window)
            window -= 1
        elif isinstance(window, timedelta):
            assert window > timedelta(0)
            csp.set_buffering_policy(xs, tick_history=window)

    # The logic below uses a number of different ways of creating the pandas data frame for efficiency.
    # An even more efficient method might be to construct the underlying block manager directly,
    # i.e. "pandas.core.internals.construction.arrays_to_mgr" with verify_integrity=False
    if csp.ticked(trigger):
        if (
            has_tindex and know_all_valid
        ):  # More efficient (2-3x) code path if we know a priori that all the times are the same
            keys = xs.keys()
            times, values = csp.items_at(xs[keys[0]], -window, None)
            idx = pandas.DatetimeIndex(times)
            arrays = {keys[0]: values}
            for k in keys[1:]:
                arrays[k] = csp.values_at(xs[k], -window, None)
            return pandas.DataFrame(arrays, columns=None, index=idx, copy=False)
        else:
            data = {}
            lengths = set()
            for k in xs.keys():
                times, values = csp.items_at(xs[k], -window, None)
                data[k] = (times, values)
                lengths.add(len(times))

            # Two different ways for performance, depending on whether we can deduce that the time index is the same
            if has_tindex and len(lengths) == 1:  # Now know that time index is the same for all series
                times, values_ = next(iter(data.values()))
                idx = pandas.DatetimeIndex(times)
                arrays = {k: values for k, (times, values) in data.items()}
                return pandas.DataFrame(arrays, columns=None, index=idx, copy=False)
            else:  # Need to do things the slow way
                series = {}
                for k, (times, values) in data.items():
                    series[k] = pandas.Series(values, index=pandas.DatetimeIndex(times))
                return pandas.DataFrame(series, columns=None, index=None, copy=False)


@csp.node
def _basket_valid(xs: {object: ts[object]}) -> ts[bool]:
    if csp.valid(xs):
        csp.make_passive(xs)
        return True
    return False


@csp.graph
def make_pandas(
    trigger: ts[object],
    data: {object: ts[object]},
    window: object,
    tindex: ts[object] = None,
    wait_all_valid: bool = True,
) -> ts[pandas.DataFrame]:
    """Graph that will generate a time series of pandas DataFrames
    :param trigger: The trigger for generation and output of the DataFrame. A new DataFrame will be produced each time
                    trigger ticks.
    :param data: A dict basket of the input time series
    :param window: An integer or timedelta representing the maximum window size (i.e. scope of the frame index).
    :param tindex: An (optional) time series on which to sample the data that goes into each frame.
                    The index of the returned DataFrames will contain the tick times of tindex.
                    This aligns the indices of the columns, improving performance.
    :param wait_all_valid: Whether to wait for all columns to be valid before including a row in the data set.
            If 'tindex' is provided, and wait_all_valid is True, the DataFrame can be constructed from the buffer numpy
            arrays with no copying. If it is False, any columns that tick after the first tick of 'tindex' (and
            are still included in the window) will be copied each time the trigger causes a new DataFrame to be generated.
    :returns: A time series of pandas DataFrames
    """
    sampled = {}
    if wait_all_valid:
        all_valid = _basket_valid(data)
        # Only trigger data frame creation
        trigger = csp.filter(all_valid, trigger)  # To make sure we don't generate empty data frames
    for k, x in data.items():
        if tindex is None:
            sampled[k] = data[k]
        else:
            sampled[k] = csp.sample(tindex, x)
        if wait_all_valid:
            sampled[k] = csp.filter(all_valid, sampled[k])
    has_tindex = tindex is not None
    return _pandas_at(trigger, sampled, window, has_tindex=has_tindex, know_all_valid=wait_all_valid)
