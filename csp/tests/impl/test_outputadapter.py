"""this test is derived from e_14_user_adapters_05 and e_14_user_adapters_06"""

import inspect
import random
import threading
import unittest
from datetime import datetime, timedelta
from json import dumps

import csp
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.outputadapter import OutputAdapter
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_output_adapter_def, py_push_adapter_def


class MyData(csp.Struct):
    symbol: str
    value: int


class MyBufferWriterAdapterImpl(OutputAdapter):
    def __init__(self, output_buffer):
        super().__init__()
        self.input_buffer = []
        self.output_buffer = output_buffer

    def start(self):
        # do this in the `start` to demonstrate opening
        # access to a resource at graph start
        self.output_buffer.clear()

    def stop(self):
        # do this in the `end` to demonstrate closing
        # access to a resource at graph stop
        data = dumps(self.input_buffer)
        self.output_buffer.append(data)

    def on_tick(self, time, value):
        self.input_buffer.append(value)


MyBufferWriterAdapter = py_output_adapter_def(
    name="MyBufferWriterAdapter",
    adapterimpl=MyBufferWriterAdapterImpl,
    input=ts["T"],
    output_buffer=list,
)


class MyAdapterManager(AdapterManagerImpl):
    def __init__(self, interval: timedelta):
        self._interval = interval
        self._counter = 0
        self._subscriptions = {}
        self._publications = {}
        self._running = False
        self._thread = None

        self._outputs = {}

    def subscribe(self, symbol):
        return _my_input_adapter(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, data: ts["T"], filename: str):
        return _my_output_adapter(self, data, filename)

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
        symbols = list(self._subscriptions.keys())
        for _ in range(10):
            # Lets pick a random symbol from the requested symbols
            symbol = symbols[random.randint(0, len(symbols) - 1)]

            data = MyData(symbol=symbol, value=self._counter)

            self._counter += 1

            for adapter in self._subscriptions[symbol]:
                # push to all the subscribers
                adapter.push_tick(data)

    def _on_tick(self, symbol, value):
        if symbol not in self._outputs:
            self._outputs[symbol] = []
        self._outputs[symbol].append("{}:{}".format(self._publications[symbol], value))


class MyInputAdapterImpl(PushInputAdapter):
    def __init__(self, manager, symbol):
        manager.register_subscription(symbol, self)
        super().__init__()


class MyOutputAdapterImpl(OutputAdapter):
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


adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
output_buffer = []


@csp.graph
def my_graph_basic():
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]

    curve = csp.curve(data=[(timedelta(seconds=1), d) for d in data], typ=object)

    MyBufferWriterAdapter(curve, output_buffer=output_buffer)


@csp.graph
def my_graph_with_manager():
    data_1 = adapter_manager.subscribe("data_1")
    data_2 = adapter_manager.subscribe("data_2")
    data_3 = adapter_manager.subscribe("data_3")

    csp.print("data_1", data_1)
    csp.print("data_2", data_2)
    csp.print("data_3", data_3)

    # pump two streams into 1 file and 1 stream into another
    adapter_manager.publish(data_1, "data_1")
    adapter_manager.publish(data_2, "data_1")
    adapter_manager.publish(data_3, "data_3")


class TestPythonOutputAdapter(unittest.TestCase):
    def test_basic(self):
        csp.run(my_graph_basic, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=False)
        self.assertEqual(
            output_buffer[0], '[{"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}]'
        )

    def test_with_manager(self):
        csp.run(my_graph_with_manager, starttime=datetime.utcnow(), endtime=timedelta(seconds=1), realtime=True)

        # assert that the adapter manager put the right things in the right place,
        # e.g. data_1 and data_2 into data_1 output, and data_3 into data_3 output
        # as per the graph above
        self.assertTrue(len(adapter_manager._outputs) > 0)
        for output_channel in adapter_manager._outputs:
            for entry in adapter_manager._outputs:
                if "symbol=data_1" in entry:
                    self.assertIn("publication_data_1", entry)
                elif "symbol=data_2" in entry:
                    # make sure data_2 inputs go into data_1 outputs
                    self.assertIn("publication_data_1", entry)
                elif "symbol=data_3" in entry:
                    self.assertIn("publication_data_3", entry)

    def test_help(self):
        # for `help` to work on output adapters, signature must be defined
        sig = inspect.signature(MyBufferWriterAdapter)
        self.assertEqual(sig.parameters["input"].annotation, ts["T"])
        self.assertEqual(sig.parameters["output_buffer"].annotation, list)
