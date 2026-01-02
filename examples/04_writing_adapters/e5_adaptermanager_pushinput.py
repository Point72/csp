"""
This example introduces the concept of an AdapterManager for realtime data. AdapterManagers are constructs that are used
when you have a shared input or output resources (ie single CSV / Parquet file, some pub/sub session, etc)
that you want to connect to once, but provide data to/from many input/output adapters (aka time series)
"""

import random
import threading
import time
from datetime import timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
from csp.utils.datetime import utc_now


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
        return MyPushAdapter(self, symbol, push_mode=push_mode)

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

        # Out driving thread, all  realtime adapters will need a separate thread of execution that
        # drives data into the engine thread
        self._running = False
        self._thread = None

    def start(self, starttime, endtime):
        """start will get called at the start of the engine run. At this point
        one would start up the realtime data source / spawn the driving thread(s) and
         subscribe to the needed data"""
        print("MyAdapterManagerImpl::start")
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """This will be called at the end of the engine run, at which point resources should be
        closed and cleaned up"""
        print("MyAdapterManagerImpl::stop")
        if self._running:
            self._running = False
            self._thread.join()

    def register_input_adapter(self, symbol, adapter):
        """Actual PushInputAdapters will self register when they are created as part of the engine
        This is the place we gather all requested input adapters and their properties
        """
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        # Keep a list of adapters by key in case we get duplicate adapters ( should be memoized in reality )
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        """This method is only used by simulated / historical adapters, for realtime we just return None"""
        return None

    def _run(self):
        """Our driving thread, in reality this will be reacting to external events, parsing the data and
        pushing it into the respective adapter
        """
        symbols = list(self._inputs.keys())
        while self._running:
            # Lets pick a random symbol from the requested symbols
            symbol = symbols[random.randint(0, len(symbols) - 1)]
            adapters = self._inputs[symbol]
            data = MyData(symbol=symbol, value=self._counter)
            self._counter += 1
            for adapter in adapters:
                adapter.push_tick(data)

            time.sleep(self._interval.total_seconds())


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time. a managed sim adapter impl will get the
# adapter manager runtime impl as its first argument
class MyPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        print(f"MyPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


MyPushAdapter = py_push_adapter_def("MyPushAdapter", MyPushAdapterImpl, ts[MyData], MyAdapterManager, symbol=str)


@csp.graph
def my_graph():
    print("Start of graph building")

    adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
    symbols = ["AAPL", "IBM", "TSLA", "GS", "JPM"]
    for symbol in symbols:
        # your data source might tick faster than the engine thread can consume it
        # push_mode can be used to buffered up tick events will get processed
        # LAST_VALUE will conflate and only tick the latest value since the last cycle
        data = adapter_manager.subscribe(symbol, csp.PushMode.LAST_VALUE)
        csp.print(symbol + " last_value", data)

        # BURST will change the timeseries type from ts[T] to ts[[T]] ( list of ticks )
        # that will tick with all values that have buffered since the last engine cycle
        data = adapter_manager.subscribe(symbol, csp.PushMode.BURST)
        csp.print(symbol + " burst", data)

        # NON_COLLAPSING will tick all events without collapsing, unrolling the events
        # over multiple engine cycles
        data = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        csp.print(symbol + " non_collapsing", data)

    print("End of graph building")


def main():
    csp.run(my_graph, starttime=utc_now(), endtime=timedelta(seconds=2), realtime=True)


if __name__ == "__main__":
    main()
