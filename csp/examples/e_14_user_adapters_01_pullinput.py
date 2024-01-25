"""
PullInputAdapter is the simplest form of an input adapter for historical data.  One instance is created
to provide data on a single timeseries.  There are use cases for this construct, though they are limited.
This is useful when feeding a single source of historical data into a single timeseries.  In most cases however,
you will likely have a single source that is processed and used to provide data to multiple inputs.  For that construct
see e_14_user_adapters_02_adaptermanager_siminput.py
"""

from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.pulladapter import PullInputAdapter
from csp.impl.wiring import py_pull_adapter_def


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time!
class MyPullAdapterImpl(PullInputAdapter):
    def __init__(self, interval: timedelta, num_ticks: int):
        print("MyPullAdapterImpl::__init__")
        self._interval = interval
        self._num_ticks = num_ticks
        self._counter = 0
        self._next_time = None
        super().__init__()

    def start(self, start_time, end_time):
        """This is called at the start of the engine, prepare your input stream here"""
        print("MyPullAdapterImpl::start")
        super().start(start_time, end_time)
        self._next_time = start_time

    def stop(self):
        """This is called at the end of the run, can shutdown / cleanup in here"""
        print("MyPullAdapterImpl::stop")

    def next(self):
        """return tuple of datetime, value of next tick, or None if no more data is available"""
        if self._counter < self._num_ticks:
            self._counter += 1
            time = self._next_time
            self._next_time += self._interval
            return time, self._counter
        return None


# MyPullAdapter is the graph-building time construct.  This is simply a representation of what the
# input adapter is and how to create it, including the Impl to use and arguments to pass into it upon construction
MyPullAdapter = py_pull_adapter_def("MyPullAdapter", MyPullAdapterImpl, ts[int], interval=timedelta, num_ticks=int)


@csp.graph
def my_graph():
    print("Start of graph building")
    data = MyPullAdapter(timedelta(seconds=1.5), num_ticks=10)
    csp.print("data", data)
    print("End of graph building")


csp.run(my_graph, starttime=datetime(2020, 12, 28))
