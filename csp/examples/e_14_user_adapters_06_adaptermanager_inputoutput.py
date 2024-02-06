"""
This is a dummy example to illustrate using an adapter manager to maintain some information across a collection
of inputs and outputs. In a real use case, you might want to multiplex a single websocket connection and both read
messages (InputAdapter) and write messages (OutputAdapter) across the same connection. To do something like this,
you need an AdapterManager to control the shared use and lifecycle of the underlying resource.

In this dummy example, the inputs are simulated data feeds (a symbol combination of string `symbol` and integer `value`),
and the outputs are print statements. However, it demonstrates the fundamental functionality of having a manager keep track
of the subscribers and publishers and manage the underlying resources accordingly.
"""

import random
import threading
import time
import typing
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.outputadapter import OutputAdapter
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_output_adapter_def, py_push_adapter_def

T = typing.TypeVar("T")


class MyData(csp.Struct):
    """This is a simple struct which mimics an inbound data feed"""

    symbol: str
    value: int


class MyAdapterManager(AdapterManagerImpl):
    """
    In this example, we do not need to separate our AdapterManager and our AdapterManagerImpl (though we could).
    Instead for brevity, we combine them into the same class, and have the `_create` method just return `self`.

    This example adapter will generate random `MyData` structs every `interval`. This simulates an upstream
    data feed, which we "connect" to only a single time. We then multiplex the results to an arbitrary
    number of subscribers via the `subscribe` method.

    We can also receive messages via the `publish` method from an arbitrary number of publishers. These messages
    are demultiplexex into a number of outputs, simulating sharing a connection to a downstream feed or responses
    to the upstream feed.
    """

    def __init__(self, interval: timedelta):
        self._interval = interval
        self._counter = 0
        self._subscriptions = {}
        self._publications = {}
        self._running = False
        self._thread = None

    def subscribe(self, symbol):
        """This method creates a new input adapter implementation via the manager."""
        return _my_input_adapter(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, data: ts["T"], symbol: str):
        """This method creates a new output adapter implementation via the manager."""
        return _my_output_adapter(self, data, symbol)

    def _create(self, engine, memo):
        # We'll avoid having a second class and make our AdapterManager and AdapterManagerImpl the same
        super().__init__(engine)
        return self

    def start(self, starttime, endtime):
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._thread.join()

        # print closing of the resources
        for name in self._publications.values():
            print("closing asset {}".format(name))

    def register_subscription(self, symbol, adapter):
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = []
        self._subscriptions[symbol].append(adapter)

    def register_publication(self, symbol):
        if symbol not in self._publications:
            self._publications[symbol] = "publication_{}".format(symbol)

    def _run(self):
        """This method runs in a background thread and generates random input events to push to the corresponding adapter"""
        symbols = list(self._subscriptions.keys())
        while self._running:
            # Lets pick a random symbol from the requested symbols
            symbol = symbols[random.randint(0, len(symbols) - 1)]

            data = MyData(symbol=symbol, value=self._counter)

            self._counter += 1

            for adapter in self._subscriptions[symbol]:
                # push to all the subscribers
                adapter.push_tick(data)

            time.sleep(self._interval.total_seconds())

    def _on_tick(self, symbol, value):
        '''This method just writes the data to the appropriate outbound "channel"'''
        print("{}:{}".format(self._publications[symbol], value))


class MyInputAdapterImpl(PushInputAdapter):
    """Our input adapter is a very simple implementation, and just
    defers its work back to the manager who is expected to deal with
    sharing a single connection.
    """

    def __init__(self, manager, symbol):
        manager.register_subscription(symbol, self)
        super().__init__()


class MyOutputAdapterImpl(OutputAdapter):
    """Similarly, our output adpter is simple as well, defering
    its functionality to the manager
    """

    def __init__(self, manager, symbol):
        manager.register_publication(symbol)
        self._manager = manager
        self._symbol = symbol
        super().__init__()

    def on_tick(self, time, value):
        self._manager._on_tick(self._symbol, value)


_my_input_adapter = py_push_adapter_def(
    name="MyInputAdapter",
    adapterimpl=MyInputAdapterImpl,
    out_type=ts[MyData],
    manager_type=MyAdapterManager,
    symbol=str,
)
_my_output_adapter = py_output_adapter_def(
    name="MyOutputAdapter", adapterimpl=MyOutputAdapterImpl, manager_type=MyAdapterManager, input=ts["T"], symbol=str
)


@csp.graph
def my_graph():
    adapter_manager = MyAdapterManager(timedelta(seconds=0.75))

    data_1 = adapter_manager.subscribe("data_1")
    data_2 = adapter_manager.subscribe("data_2")
    data_3 = adapter_manager.subscribe("data_3")

    csp.print("data_1", data_1)
    csp.print("data_2", data_2)
    csp.print("data_3", data_3)

    # pump two streams into 1 output and 1 stream into another
    adapter_manager.publish(data_1, "data_1")
    adapter_manager.publish(data_2, "data_1")
    adapter_manager.publish(data_3, "data_3")


if __name__ == "__main__":
    csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=True)
