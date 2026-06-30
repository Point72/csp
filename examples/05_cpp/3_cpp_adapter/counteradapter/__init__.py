"""
Counter Adapter - Python module for the Counter adapter example

This module demonstrates how to create a Python wrapper around C++ adapters.
It provides a clean Python interface to the CounterAdapterManager,
CounterInputAdapter, and CounterOutputAdapter.

Usage:
    from counteradapter import CounterAdapterManager

    @csp.graph
    def my_graph():
        counter_mgr = CounterAdapterManager(interval_ms=100)
        data = counter_mgr.subscribe()
        counter_mgr.publish(data)
        return data

    result = csp.run(my_graph, starttime=datetime.now(), endtime=timedelta(seconds=5))
"""

import csp
from csp.impl.wiring import input_adapter_def, output_adapter_def

# Import the C++ extension module
from . import _counteradapter


class CounterAdapterManager:
    """
    A simple example adapter manager that generates sequential counter values.

    This demonstrates the basic pattern for creating a Python wrapper around
    a C++ AdapterManager.

    Args:
        interval_ms: Interval between counter ticks in milliseconds (default: 1000)
        max_count: Maximum count before stopping, 0 for unlimited (default: 0)
    """

    def __init__(self, interval_ms: int = 1000, max_count: int = 0):
        self._properties = {
            "interval_ms": interval_ms,
            "max_count": max_count,
        }

    def _create(self, engine, memo):
        """Create the C++ adapter manager."""
        return _counteradapter._counter_adapter_manager(engine, self._properties)

    def subscribe(self) -> csp.ts[int]:
        """
        Subscribe to counter values.

        Returns a time series of integer counter values that tick at the
        configured interval.

        Returns:
            csp.ts[int]: Time series of counter values
        """
        return _counter_input_adapter(self, typ=int, properties={}, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, data: csp.ts[int]):
        """
        Publish values to the output adapter.

        This will log the values to stdout.

        Args:
            data: Time series of integer values to publish
        """
        _counter_output_adapter(self, data, typ=int, properties={})


_counter_input_adapter = input_adapter_def(
    "counter_input_adapter",
    _counteradapter._counter_input_adapter,
    csp.ts["T"],
    CounterAdapterManager,
    typ="T",
    properties=dict,
)

_counter_output_adapter = output_adapter_def(
    "counter_output_adapter",
    _counteradapter._counter_output_adapter,
    CounterAdapterManager,
    input=csp.ts["T"],
    typ="T",
    properties=dict,
)


__all__ = ["CounterAdapterManager"]
