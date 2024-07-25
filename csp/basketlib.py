from datetime import timedelta
from typing import Dict, List, TypeVar

import csp
from csp import ts
from csp.lib import _cspbasketlibimpl

__all__ = ["sync", "sync_dict", "sync_list"]
T = TypeVar("T")
K = TypeVar("K")
Y = TypeVar("Y")


@csp.node(cppimpl=_cspbasketlibimpl._sync_list)
def sync_list(x: List[ts["T"]], threshold: timedelta, output_incomplete: bool = True) -> csp.OutputBasket(
    List[ts["T"]], shape_of="x"
):
    with csp.alarms():
        a_end = csp.alarm(bool)

    with csp.state():
        s_current = {}
        s_alarm_handle = None

    if csp.ticked(x):
        if not s_alarm_handle:
            s_alarm_handle = csp.schedule_alarm(a_end, threshold, True)
        s_current.update(x.tickeditems())

    if csp.ticked(a_end) or len(s_current) == len(x):
        if len(s_current) == len(x) or output_incomplete:
            csp.output(s_current)
        if s_alarm_handle:
            csp.cancel_alarm(a_end, s_alarm_handle)
            s_alarm_handle = None
        s_current = {}


@csp.graph
def sync_dict(x: Dict["K", ts["T"]], threshold: timedelta, output_incomplete: bool = True) -> csp.OutputBasket(
    Dict["K", ts["T"]], shape_of="x"
):
    values = list(x.values())
    synced = sync_list(values, threshold, output_incomplete)
    return {k: v for k, v in zip(x.keys(), synced)}


def sync(x, threshold: timedelta, output_incomplete: bool = True):
    if isinstance(x, list):
        return sync_list(x, threshold, output_incomplete)
    elif isinstance(x, dict):
        return sync_dict(x, threshold, output_incomplete)
    raise ValueError(f"Input must be list or dict basket, got: {type(x)}")


@csp.node(cppimpl=_cspbasketlibimpl._sample_list)
def sample_list(trigger: ts["Y"], x: List[ts["T"]]) -> csp.OutputBasket(List[ts["T"]], shape_of="x"):
    """will return valid items in x on trigger"""
    with csp.start():
        csp.make_passive(x)

    if csp.ticked(trigger):
        result = {k: v for k, v in x.validitems()}
        if result:
            return result


@csp.graph()
def sample_dict(trigger: ts["Y"], x: Dict["K", ts["T"]]) -> csp.OutputBasket(Dict["K", ts["T"]], shape_of="x"):
    """will return valid items in x on trigger"""
    values = list(x.values())
    sampled_values = sample_list(trigger, values)
    return {key: value for key, value in zip(x.keys(), sampled_values)}


def sample_basket(trigger, x):
    """will return valid items in x on trigger"""
    if isinstance(x, list):
        return sample_list(trigger, x)
    elif isinstance(x, dict):
        return sample_dict(trigger, x)
    raise ValueError(f"Input must be a list or dict basket, got: {type(x)}")
