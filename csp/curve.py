import copy
from datetime import timedelta
from typing import Union

import numpy as np
import pytz

from csp import null_ts
from csp.impl.__cspimpl import _cspimpl
from csp.impl.pulladapter import PullInputAdapter
from csp.impl.types.common_definitions import PushMode
from csp.impl.types.tstype import ts
from csp.impl.wiring import input_adapter_def, py_pull_adapter_def


class Curve(PullInputAdapter):
    def __init__(self, typ, data):
        """data should be a list of tuples of ( datetime, value ) or ( timedelta, value )"""
        self._data = data
        self._index = 0
        super().__init__()

    def start(self, start_time, end_time):
        if isinstance(self._data[0][0], timedelta):
            self._data = copy.copy(self._data)
            for idx, data in enumerate(self._data):
                self._data[idx] = (start_time + data[0], data[1])
        elif self._data[0][0].tzinfo is not None:
            for idx, data in enumerate(self._data):
                self._data[idx] = (data[0].astimezone(pytz.UTC).replace(tzinfo=None), data[1])

        while self._index < len(self._data) and self._data[self._index][0] < start_time:
            self._index += 1

        super().start(start_time, end_time)

    def next(self):
        if self._index < len(self._data):
            time, value = self._data[self._index]
            if time <= self._end_time:
                self._index += 1
                return time, value
        return None


_curve = py_pull_adapter_def("csp.curve", Curve, ts["T"], typ="T", data=list)
_npcurve = input_adapter_def(
    "csp.curve", _cspimpl._npcurve, ts["T"], typ="T", datetimes=np.ndarray, values=np.ndarray, memoize=False
)


def curve(typ: type, data: Union[list, tuple], push_mode: PushMode = PushMode.NON_COLLAPSING):
    if isinstance(data, tuple):
        if len(data) != 2 or not all(isinstance(x, np.ndarray) for x in data):
            raise ValueError("for numpy curves, must pass tuple of two ndarrays as data")
        if len(data[0]) != len(data[1]):
            raise ValueError("ndarrays passed to csp.curve must be of equal length")

        if len(data[0]) == 0:
            return null_ts(typ)
        return _npcurve(typ=typ, datetimes=data[0], values=data[1], push_mode=push_mode)

    if len(data) == 0:
        return null_ts(typ)
    return _curve(typ=typ, data=data, push_mode=push_mode)
