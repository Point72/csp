"""
This example introduces the concept of an AdapterManager. AdapterManagers are constructs that are used
when you have a shared input or output resources (ie single CSV / Parquet file, some pub/sub session, etc)
that you want to connect to once, but provide data to/from many input/output adapters (aka time series)
"""

import random
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
from csp.impl.wiring import py_managed_adapter_def


class MyData(csp.Struct):
    symbol: str
    value: int


# This object represents our AdapterManager at graph time. It describes the manager's properties
# and will be used to create the actual impl when its time to build the engine
class MyAdapterManager:
    def __init__(self, interval: timedelta):
        """
        Normally one would pass properties of the manager here, ie filename,
        message bus, etc
        """
        print("MyAdapterManager::__init__")
        self._interval = interval

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        """User facing API to subscribe to a timeseries stream from this adapter manager"""
        # This will return a graph-time timeseries edge representing and edge from this
        # adapter manager for the given symbol / arguments
        return MyManagedSimAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        """This method will get called at engine build time, at which point the graph time manager representation
        will create the actual impl that will be used for runtime
        """
        print("MyAdapterManager::_create")
        # Normally you would pass the arguments down into the impl here
        return MyAdapterManagerImpl(engine, self._interval)


# This is the actual manager impl that will be created and executed during runtime
class MyAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, interval):
        print("MyAdapterManagerImpl::__init__")
        super().__init__(engine)

        # These are just used to simulate a data source
        self._interval = interval
        self._counter = 0

        # We will keep track of requested input adapters here
        self._inputs = {}

    def start(self, starttime, endtime):
        """Over here one would open up the resource, build up a query for the requested inputs
        and seek to starttime to prepare for processing"""
        print("MyAdapterManagerImpl::start")

    def stop(self):
        """This will be called at the end of the engine run, at which point resources should be
        closed and cleaned up"""
        print("MyAdapterManagerImpl::stop")

    def register_input_adapter(self, symbol, adapter):
        """Actual ManagedPullInput adapters will self register when they are created as part of the engine
        This is the place we gather all requested input adapters and their properties
        """
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        # Keep a list of adapters by key in case we get duplicate adapters ( should be memoized in reality )
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        """After start is called, process_next_sim_timeslice will be called repeatedly
        to process the next available timestamp from the data source. Every call to this method
        should process all "rows" for the given timestamp.
        For every tick that applies to an input, we push the tick into the adapter.
        This method should return the datetime of the next even in the data, or None if there is no data left.
        First call will be for "starttime"
        """

        # Generate random data, simulate some number of rows per timeslice
        num_rows = random.randint(0, 10)

        symbols = list(self._inputs.keys())
        while num_rows > 0:
            # Lets pick a random symbol from the requested symbols
            symbol = symbols[random.randint(0, len(symbols) - 1)]
            adapters = self._inputs[symbol]
            data = MyData(symbol=symbol, value=self._counter)
            self._counter += 1
            for adapter in adapters:
                adapter.push_tick(data)

            num_rows -= 1

        return now + self._interval


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time. a managed sim adapter impl will get the
# adapter manager runtime impl as its first argument
class MyManagedSimAdapterImpl(ManagedSimInputAdapter):
    def __init__(self, manager_impl, symbol):
        print(f"MyManagedSimAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__(MyData, None)


# Note that the push_mode argument is implicitly added as an argument to the adapter
MyManagedSimAdapter = py_managed_adapter_def(
    "MyManagedSimAdapter", MyManagedSimAdapterImpl, ts[MyData], MyAdapterManager, symbol=str
)


@csp.graph
def my_graph():
    print("Start of graph building")

    adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
    symbols = ["AAPL", "IBM", "TSLA", "GS", "JPM"]
    for symbol in symbols:
        # If your data source happens to tick multiple times on the same timeseries
        # at the same time, then push mode will determine how the duplicate time ticks
        # will tick.
        # LAST_VALUE will conflate and only tick the last value of a given timestamp
        data = adapter_manager.subscribe(symbol, csp.PushMode.LAST_VALUE)
        csp.print(symbol + " last_value", data)

        # BURST will change the timeseries type from ts[T] to ts[[T]] ( list of ticks )
        # that will tick with all values in a given timestamp as a list
        data = adapter_manager.subscribe(symbol, csp.PushMode.BURST)
        csp.print(symbol + " burst", data)

        # NON_COLLAPSING will tick all events without collapsing, unrolling the events
        # over multiple engine cycles
        data = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        csp.print(symbol + " non_collapsing", data)

    print("End of graph building")


def main():
    csp.run(my_graph, starttime=datetime(2020, 12, 28), endtime=timedelta(seconds=10))


if __name__ == "__main__":
    main()
