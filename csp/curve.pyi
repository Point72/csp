"""Type stubs for csp curve function."""

from typing import List, Tuple, Type, TypeVar, Union

import numpy as np

from csp.impl.types.common_definitions import PushMode
from csp.impl.types.tstype import TsType

T = TypeVar("T")

def curve(
    typ: Type[T],
    data: Union[List[Tuple], Tuple[np.ndarray, np.ndarray]],
    push_mode: PushMode = ...,
) -> TsType[T]:
    """
    Create a time series from static data.

    Args:
        typ: Type of the time series values
        data: Either:
            - List of (datetime/timedelta, value) tuples
            - Tuple of two numpy arrays (datetimes, values)
        push_mode: How to handle multiple values at same timestamp

    Returns:
        Time series that ticks according to the provided data

    Example:
        from datetime import datetime, timedelta

        # Using datetime tuples
        data = [
            (datetime(2024, 1, 1, 9, 30), 100.0),
            (datetime(2024, 1, 1, 9, 31), 101.0),
        ]
        prices = csp.curve(float, data)

        # Using timedelta (relative to start time)
        data = [
            (timedelta(seconds=0), 100.0),
            (timedelta(seconds=1), 101.0),
        ]
        prices = csp.curve(float, data)

        # Using numpy arrays
        import numpy as np
        times = np.array([...], dtype='datetime64[ns]')
        values = np.array([100.0, 101.0])
        prices = csp.curve(float, (times, values))
    """
    ...
