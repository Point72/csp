"""
PushInputAdapter is the simplest form of an input adapter for real-time data. One instance is created
to provide data on a single timeseries. There are use cases for this construct, though they are limited.
This is useful when feeding a single source of data into a single timeseries. In most cases however,
you will likely have a single source that is processed and used to provide data to multiple inputs. For that construct
see e5_adaptermanager_pushinput.py
"""

import threading
import time
from datetime import timedelta

import csp
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
from csp.utils.datetime import utc_now


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time!
class MyPushAdapterImpl(PushInputAdapter):
    def __init__(self, interval):
        print("MyPushAdapterImpl::__init__")
        self._interval = interval
        self._thread = None
        self._running = False

    def start(self, starttime, endtime):
        """start will get called at the start of the engine, at which point the push
        input adapter should start its thread that will push the data onto the adapter. Note
        that push adapters will ALWAYS have a separate thread driving ticks into the csp engine thread
        """
        print("MyPushAdapterImpl::start")
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """stop will get called at the end of the run, at which point resources should
        be cleaned up
        """
        print("MyPushAdapterImpl::stop")
        if self._running:
            self._running = False
            self._thread.join()

    def _run(self):
        counter = 0
        while self._running:
            self.push_tick(counter)
            counter += 1
            time.sleep(self._interval.total_seconds())


# MyPushAdapter is the graph-building time construct. This is simply a representation of what the
# input adapter is and how to create it, including the Impl to create and arguments to pass into it
MyPushAdapter = py_push_adapter_def("MyPushAdapter", MyPushAdapterImpl, ts[int], interval=timedelta)


@csp.graph
def my_graph():
    # At this point we create the graph-time representation of the input adapter. This will be converted
    # into the impl once the graph is done constructing and the engine is created in order to run
    print("Start of graph building")
    data = MyPushAdapter(timedelta(seconds=1))
    csp.print("data", data)
    print("End of graph building")


def main():
    csp.run(my_graph, realtime=True, starttime=utc_now(), endtime=timedelta(seconds=2))


if __name__ == "__main__":
    main()
