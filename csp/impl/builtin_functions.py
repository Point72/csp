from datetime import datetime, timedelta
from typing import Any, TypeVar, Union

from csp.impl.constants import UNSET
from csp.impl.types.autogen_types import TimeIndexPolicy
from csp.impl.types.common_definitions import DuplicatePolicy, Outputs
from csp.impl.types.tstype import GenericTSTypes

ALL_CSP_BUILTIN_FUNCS = {}
T = TypeVar("T")


def csp_builtin(func):
    ALL_CSP_BUILTIN_FUNCS[func.__name__] = func
    return func


@csp_builtin
def num_ticks(input: GenericTSTypes["T"].TS_TYPE) -> int:
    """
    :param input: The time series for which the number of ticks should be returned
    :return: A number of ticks of the input (note, it doesn't mean that all those values can be retrieved
    from the history, since the buffer is not guaranteed to be long enough)
    """
    raise RuntimeError("Unexpected use of csp.num_ticks, csp.num_ticks can only be used inside a node")


@csp_builtin
def ticked(*ts_or_basket: GenericTSTypes["T"].TS_OR_BASKET_TYPE) -> bool:
    """Check if the given ts/basked ticked

    Usage:
    @csp.node
    def my_node(series1 : ts['T'], series2 : ts['T'], series3 : ts['T']):
        if csp.ticked(series1, series2):
            # will get here if series1 OR series2 ticked
            pass
        if csp.ticked(series3):
            # will get here if series3 ticked
            pass

    :param ts_or_basket:
    """
    raise RuntimeError("Unexpected use of csp.ticked, csp.ticked can only be used inside a node")


@csp_builtin
def valid(*ts_or_basket: GenericTSTypes["T"].TS_OR_BASKET_TYPE) -> bool:
    """Check if the given ts/basked have valid values (ticked at least once)

    Usage:
    @csp.node
    def my_node(series1 : ts['T'], series2 : ts['T'], series3 : ts['T']):
        if csp.valid(series1, series2):
            # will get here if series1 AND series2 are valid
            # NOTE: It's common to combine checks of valid with ticked,
            # we get here on every tick of series1, series2 or series3 after
            # series1 and series2  become valid, often this is undesired and
            the "if" condition often needs to check ticked and valid
            pass
    :param ts_or_basket:
    """
    raise RuntimeError("Unexpected use of csp.valid, csp.valid can only be used inside a node")


@csp_builtin
def make_passive(ts_or_basket: GenericTSTypes["T"].TS_OR_BASKET_TYPE) -> bool:
    """Make the given ts or basket passive - won't trigger ticks in the output node

    Usage:
    @csp.node
    def my_node(series1 : ts['T'], series2 : ts['T']):
        with __start__():
            csp.make_passive(series1)

        # DO WHATEVER IS REQUIRED HERE
        # The code here will be executed only on series2 ticks since series1
        # is marked as passive and won't trigger execution

    :param ts_or_basket:
    """
    raise RuntimeError("Unexpected use of csp.make_passive, csp.make_passive can only be used inside a node")


@csp_builtin
def make_active(ts_or_basket: GenericTSTypes["T"].TS_OR_BASKET_TYPE) -> bool:
    """Make the given ts or basket active - revert the make_passive_call

    Usage:
    @csp.node
    def my_node(series1 : ts['T'], series2 : ts['T']):
        with __start__():
            csp.make_passive(series1)

        if(some_event_happened(series1, series2)):
            # From now on, the node will be called on ticks of series1 and series2
            csp.make_active(series1)

    :param ts_or_basket:
    """
    raise RuntimeError("Unexpected use of csp.make_active, csp.make_active can only be used inside a node")


@csp_builtin
def value_at(
    series: GenericTSTypes["T"].TS_TYPE,
    index_or_time: Union[int, timedelta, datetime, None] = 0,
    duplicate_policy: int = DuplicatePolicy.LAST_VALUE,
    default: Any = UNSET,
):
    """Get value of the time series at a given index or time delta
    :param series: The series from which the value should be retrieved
    :param index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values.
    :param duplicate_policy: The policy of how duplicate values should be handled, for now only LAST_VALUE is
    supported.
    :param default: The default value to be returned if the given index/time is out of bounds. Note, None means that
    there is no default. To avoid exception on missing value, explicit None arg must be provided and this value will
    be returned if no value found
    :return: The value at the given index or time

    Examples:
    Consider a time series x that has the following ticks: (09:30, 0), (09:31, 1), (09:31, 2), (09:33, 3)
    Assume the the current time is 2020/01/01 09:33:00
    csp.value_at(x) # return the last value: 3
    csp.value_at(x, 0) # return the last value: 3
    csp.value_at(x, -1) # return the value from previous tick: 2
    csp.value_at(x, -3) # return the value 3 ticks ago: 0
    csp.value_at(x, -4) # raises an exception - no ticks exist 4 ticks ago
    csp.value_at(x, -4, default=-1) # return the default value since no ticks exist 4 ticks ago: -1
    csp.value_at(x, -4, default=None) # return the default value since no ticks exist 4 ticks ago: None
    csp.value_at(x, timedelta(minutes=-0.5) # return the value half a minute ago: 2
    csp.value_at(x, timedelta(minutes=-1)) # return the value a minute ago: 2
    csp.value_at(x, timedelta(minutes=-2)) # return the value a 2 minutes ago: 1
    csp.value_at(x, datetime(2020,1,1, 9, 30)) # return the value at 09:30 : 0
    """
    raise RuntimeError("Unexpected use of csp.value_at, csp.value_at can only be used inside a node")


@csp_builtin
def time_at(
    series: GenericTSTypes["T"].TS_TYPE,
    index_or_time: Union[int, timedelta, datetime, None] = 0,
    duplicate_policy: int = DuplicatePolicy.LAST_VALUE,
    default: Any = UNSET,
):
    """Get timestamp of the time series at a given index or time delta
    :param series: The series from which the timestamp should be retrieved
    :param index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values.
    :param duplicate_policy: The policy of how duplicate values should be handled, for now only LAST_VALUE is
    supported.
    :param default: The default value to be returned if the given index/time is out of bounds. Note, None means that
    there is no default. To avoid exception on missing value, explicit None arg must be provided and this value will
    be returned if no value found
    :return: The datetime timestamp at the given index or time

    Examples:
    Consider a time series x that has the following ticks: (09:30, 0), (09:31, 1), (09:31, 2), (09:33, 3)
    Assume the the current time is 2020/01/01 09:33:00
    csp.timestamp_at(x) # return the last value: 09:33
    csp.timestamp_at(x, 0) # return the last value: 09:33
    csp.timestamp_at(x, -1) # return the value from previous tick: 09:31
    csp.timestamp_at(x, -3) # return the value 3 ticks ago: 09:30
    csp.timestamp_at(x, -4) # raises an exception - no ticks exist 4 ticks ago
    csp.timestamp_at(x, -4, default=datetime.now()) # return the default value since no ticks exist 4 ticks ago: datetime.now()
    csp.timestamp_at(x, -4, default=None) # return the default value since no ticks exist 4 ticks ago: None
    csp.timestamp_at(x, timedelta(minutes=-0.5) # return the value half a minute ago: 09:31
    csp.timestamp_at(x, timedelta(minutes=-1)) # return the value a minute ago: 09:31
    csp.timestamp_at(x, timedelta(minutes=-2)) # return the value a 2 minutes ago: 09:31
    csp.timestamp_at(x, datetime(2020,1,1, 9, 30)) # return the value at 09:30 : 09:30
    """
    raise RuntimeError("Unexpected use of csp.timestamp_at, csp.timestamp_at can only be used inside a node")


@csp_builtin
def item_at(
    series: GenericTSTypes["T"].TS_TYPE,
    index_or_time: Union[int, timedelta, datetime, None] = 0,
    duplicate_policy: int = DuplicatePolicy.LAST_VALUE,
    default: Any = UNSET,
):
    """Get a tuple containing tuple (timestamp, value) it's the same as (timetamp_at(...), value_at(...)) but faster
    :param series: The series from which the values should be retrieved
    :param index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values.
    :param duplicate_policy: The policy of how duplicate values should be handled, for now only LAST_VALUE is
    supported.
    :param default: The default value to be returned if the given index/time is out of bounds. Note, None means that
    there is no default. To avoid exception on missing value, explicit None arg must be provided and this value will
    be returned if no value found
    :return: The the tuple(datetime,value) at the given index or time
    """
    raise RuntimeError("Unexpected use of csp.item_at, csp.item_at can only be used inside a node")


@csp_builtin
def values_at(
    series: GenericTSTypes["T"].TS_TYPE,
    start_index_or_time: Union[int, timedelta, datetime, None] = None,
    end_index_or_time: Union[int, timedelta, datetime, None] = None,
    start_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
    end_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
):
    """Get values of the time series between given start and end indices or time deltas
    :param series: The series from which the value should be retrieved
    :param start_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get values "from the start"
    :param end_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get values "until the end"
    :param start_index_policy: TimeIndexPolicy enum to indicate whether the start time should be included, excluded, or forced.
    :param end_index_policy: TimeIndexPolicy enum to indicate whether the end time should be included, excluded, or forced.
    :return: The values between the given indices or times as np.ndarray, in ascending (time) order.

    Examples:
    Consider a time series x that has the following ticks: (09:30, 0), (09:31, 1), (09:31, 2), (09:33, 3)
    Assume the the current time is 2020/01/01 09:34:00
    csp.values_at( x, -3, -1, False, False, True ) # np.ndarray( [ 0, 1, 2 ] )
    """
    raise RuntimeError("Unexpected use of csp.values_at, csp.values_at can only be used inside a node")


@csp_builtin
def times_at(
    series: GenericTSTypes["T"].TS_TYPE,
    start_index_or_time: Union[int, timedelta, datetime, None] = None,
    end_index_or_time: Union[int, timedelta, datetime, None] = None,
    start_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
    end_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
):
    """Get timestamps of the time series between given start and end indices or time deltas
    :param series: The series from which the timestamp should be retrieved
    :param start_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get times "from the start"
    :param end_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get times "until the end"
    :param start_index_policy: TimeIndexPolicy enum to indicate whether the start time should be included, excluded, or forced.
    :param end_index_policy: TimeIndexPolicy enum to indicate whether the end time should be included, excluded, or forced.
    :return: The values between the given indices or times as np.ndarray, in ascending (time) order.

    Examples:
    Consider a time series x that has the following ticks: (09:30, 0), (09:31, 1), (09:31, 2), (09:33, 3)
    Assume the the current time is 2020/01/01 09:34:00
    csp.times_at( x, -3, -1, False, False, True ) # np.ndarray( [ (09:30), (09:31), (09:31) ] )
    """
    raise RuntimeError("Unexpected use of csp.times_at, csp.times_at can only be used inside a node")


@csp_builtin
def items_at(
    series: GenericTSTypes["T"].TS_TYPE,
    start_index_or_time: Union[int, timedelta, datetime, None] = None,
    end_index_or_time: Union[int, timedelta, datetime, None] = None,
    start_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
    end_index_policy: TimeIndexPolicy = TimeIndexPolicy.INCLUSIVE,
):
    """Get a tuple containing tuple (timestamps, values) it's the same as (times_at(...), values_at(...)) but faster
    :param series: The series from which the timestamp should be retrieved
    :param start_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get items "from the start"
    :param end_index_or_time: Can be an integer index, timedelta or datetime. Both index and timedelta are expected to
    be non positive values. Set to None to get items "until the end"
    :param start_index_policy: TimeIndexPolicy enum to indicate whether the start time should be included, excluded, or forced.
    :param end_index_policy: TimeIndexPolicy enum to indicate whether the end time should be included, excluded, or forced.
    :return: The items between the given indices or times as a tuple of np.ndarray, in ascending (time) order.
    """
    raise RuntimeError("Unexpected use of csp.items_at, csp.items_at can only be used inside a node")


@csp_builtin
def set_buffering_policy(
    ts_or_basket: GenericTSTypes["T"].TS_OR_BASKET_TYPE, tick_count: int = None, tick_history: timedelta = None
):
    """Set the buffering window on the given timeseries or basket of timeseries.  By default only last tick is kept.
    at least one of tick_count or tick_history is required, though both can be supplied
    :param ts_or_basket: ts or basket input
    :param tick_count: number of ticks to keep in history
    :param tick_history: how much time to keep in the buffer
    """
    raise RuntimeError(
        "Unexpected use of csp.set_buffering_policy, csp.set_buffering_policy can only be used inside a node"
    )


@csp_builtin
def alarm(typ: "T") -> GenericTSTypes["T"].TS_TYPE:
    """Initialized an alarm event.
    :param when: timedelta or datetime when to tick the alarm.  timedelta is offset from csp.now(). Use `None` for "unscheduled"
    :param value: value to tick on alarm input at scheduled time
    """
    raise RuntimeError("Unexpected use of csp.alarm, csp.alarm can only be used inside a node")


@csp_builtin
def schedule_alarm(series: GenericTSTypes["T"].TS_TYPE, when: Union[datetime, timedelta], value: Any):
    """Schedule alarm event.  May be called multiple times to schedule multiple alarms
    :param series: alarm input to schedule on
    :param when: timedelta or datetime when to tick the alarm.  timedelta is offset from csp.now().
    :param value: value to tick on alarm input at scheduled time
    """
    raise RuntimeError("Unexpected use of csp.schedule_alarm, csp.schedule_alarm can only be used inside a node")


@csp_builtin
def now() -> datetime:
    """Returns the current engine time as a tz-less UTC datetime"""
    raise RuntimeError("Unexpected use of csp.now, csp.now can only be used inside a node")


@csp_builtin
def remove_dynamic_key(basket: GenericTSTypes["T"].TS_TYPE, key: Any):
    """Remove the given key from the given dynamic basket output
    :param basket: dynamic basket to remove from
    :param key:    key to remove
    """
    raise RuntimeError(
        "Unexpected use of csp.remove_dynamic_key, csp.remove_dynamic_key can only be used inside a node"
    )


@csp_builtin
def engine_start_time() -> datetime:
    """Returns the engine run start time (can be used both in nodes and graphs)"""
    from csp.impl.wiring import GraphRunInfo

    return GraphRunInfo.get_cur_run_times_info().starttime


@csp_builtin
def engine_end_time() -> datetime:
    """Returns the engine run end time (can be used both in nodes and graphs)"""
    from csp.impl.wiring import GraphRunInfo

    return GraphRunInfo.get_cur_run_times_info().endtime


@csp_builtin
def is_configured_realtime() -> bool:
    """Returns whether the graph is configured to run in realtime mode"""
    from csp.impl.wiring import GraphRunInfo

    return GraphRunInfo.get_cur_run_times_info().is_realtime


@csp_builtin
def set_capture_cpp_backtrace(value: bool = True):
    """Sets global setting on whether c++ exception should capture the c++ backtrace"""
    from csp.impl.__cspimpl import _cspimpl

    _cspimpl.set_capture_cpp_backtrace(value)


@csp_builtin
def cancel_alarm(*args, **kwargs):
    raise RuntimeError("Unexpected use of csp.cancel_alarm, csp.cancel_alarm can only be used inside a node")


@csp_builtin
def output(*args, **kwargs) -> Outputs:
    raise RuntimeError("Unexpected use of csp.output, possibly using outside of @graph and @node?")


@csp_builtin
def __outputs__(*args, **kwargs):
    raise RuntimeError("Unexpected use of __outputs__, possibly using outside of @graph and @node?")


@csp_builtin
def __return__(**kwargs):
    raise RuntimeError("Unexpected use of __return__, possibly using outside of @graph and @node?")


@csp_builtin
def __state__(**kwargs):
    raise RuntimeError("Unexpected use of __state__, possibly using outside of @node?")


@csp_builtin
def __alarms__(**kwargs):
    raise RuntimeError("Unexpected use of __alarms__, possibly using outside of @node?")


@csp_builtin
def __start__(**kwargs):
    raise RuntimeError("Unexpected use of __start__, possibly using outside of @node?")


@csp_builtin
def __stop__(**kwargs):
    raise RuntimeError("Unexpected use of __stop__, possibly using outside of @node?")


@csp_builtin
def state(**kwargs):
    raise RuntimeError("Unexpected use of state, possibly using outside of @node?")


@csp_builtin
def alarms(**kwargs):
    raise RuntimeError("Unexpected use of alarms, possibly using outside of @node?")


@csp_builtin
def start(**kwargs):
    raise RuntimeError("Unexpected use of start, possibly using outside of @node?")


@csp_builtin
def stop(**kwargs):
    raise RuntimeError("Unexpected use of stop, possibly using outside of @node?")


@csp_builtin
def engine_stats(*args, **kwargs):
    raise RuntimeError("Unexpected use of csp.engine_stats, csp.engine_stats can only be used inside a node")


__all__ = [
    "ALL_CSP_BUILTIN_FUNCS",
    "DuplicatePolicy",
    "TimeIndexPolicy",
    "UNSET",
] + list(ALL_CSP_BUILTIN_FUNCS.keys())

CSP_BUILTIN_CONTEXT_DICT = {v: globals()[v] for v in __all__}
