## Table of Contents

- [Table of Contents](#table-of-contents)
- [Output Adapters](#output-adapters)
  - [OutputAdapter - Python](#outputadapter---python)
  - [OutputAdapter - C++](#outputadapter---c)
  - [OutputAdapter with Manager](#outputadapter-with-manager)
  - [InputOutputAdapter - Python](#inputoutputadapter---python)

## Output Adapters

Output adapters are used to define graph outputs, and they differ from input adapters in a number of important ways.
Output adapters also differ from terminal nodes, e.g. regular `csp.node` instances that do not define outputs, and instead consume and emit their inputs inside their `csp.ticked`  blocks.

For many use cases, it will be sufficient to omit writing an output adapter entirely.
Consider the following example of a terminal node that writes an input dictionary timeseries to a file.

```python
@csp.node
def write_to_file(x: ts[Dict], filename: str):
    if csp.ticked(x):
        with open(filename, "a") as fp:
            fp.write(json.dumps(x))
```

This is a perfectly fine node, and serves its purpose.
Unlike input adapters, output adapters do not need to differentiate between *historical* and *realtime* mode.
Input adapters drive the execution of the graph, whereas output adapters are reactive to their input nodes and subject to the graph's execution.

However, there are a number of reasons why you might want to define an output adapter instead of using a vanilla node.
The most important of these is when you want to share resources across a number of output adapters (e.g. with a Manager), or between an input and an output node, e.g. reading data from a websocket, routing it through your CSP graph, and publishing data *to the same websocket connection*.
For most use cases, a vanilla CSP node will suffice, but let's explore some anyway.

### OutputAdapter - Python

To write a Python based OutputAdapter one must write a class that derives from `csp.impl.outputadapter.OutputAdapter`.
The derived type should define the method:

- `def on_tick(self, time: datetime, value: object)`: this will be called when the input to the output adapter ticks.

The OutputAdapter that you define will be used as the runtime *--impl–-*.  You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a `csp.impl.wiring.py_output_adapter_def`.
The `py_output_adapter_def` creates a *--graph--* time representation of your adapter:

**def py_output_adapter_def(name, adapterimpl, \*\*kwargs)**

- **`name`**: string name for the adapter
- **`adapterclass`**: a derived implementation of `csp.impl.outputadapter.OutputAdapter`
- **`kwargs`**: \*\*kwargs here be passed through as arguments to the OutputAdapter implementation

Note that the `**kwargs` passed to py_output_adapter_def should be the names and types of the variables, like `arg1=type1, arg2=type2`.
These are the names of the kwargs that the returned output adapter will take and pass through to the OutputAdapter implementation, and the types expected for the values of those args.

Here is a simple example of the same filewriter from above:

```python
from csp.impl.outputadapter import OutputAdapter
from csp.impl.wiring import py_output_adapter_def
from csp import ts
import csp
from json import dumps
from datetime import datetime, timedelta


class MyFileWriterAdapterImpl(OutputAdapter):
    def __init__(self, filename: str):
        super().__init__()
        self._filename = filename

    def start(self):
        self._fp = open(self._filename, "a")

     def stop(self):
        self._fp.close()

    def on_tick(self, time, value):
        self._fp.write(dumps(value) + "\n")


MyFileWriterAdapter = py_output_adapter_def(
    name='MyFileWriterAdapter',
    adapterimpl=MyFileWriterAdapterImpl,
    input=ts['T'],
    filename=str,
)
```

Now our adapter can be called in graph code:

```python
@csp.graph
def my_graph():
    curve = csp.curve(
        data=[
            (timedelta(seconds=0), {"a": 1, "b": 2, "c": 3}),
            (timedelta(seconds=1), {"a": 1, "b": 2, "c": 3}),
            (timedelta(seconds=1), {"a": 1, "b": 2, "c": 3}),
        ],
        typ=object,
   )

    MyFileWriterAdapter(curve, filename="testfile.jsonl")
```

As explained above, we could also do this via single node (this is probably the best version between the three):

```python
@csp.node
def dump_json(data: ts['T'], filename: str):
    with csp.state():
        s_file=None
    with csp.start():
        s_file = open(filename, "w")
    with csp.stop():
        s_file.close()
    if csp.ticked(data):
        s_file.write(json.dumps(data) + "\n")
        s_file.flush()
```

### OutputAdapter - C++

TODO

### OutputAdapter with Manager

Adapter managers function the same way for output adapters as for input adapters, i.e. to manage a single shared resource from the manager across a variety of discrete output adapters.

### InputOutputAdapter - Python

As a as last example, lets tie everything together and implement a managed push input adapter combined with a managed output adapter.
This example is available in [e7_adaptermanager_inputoutput.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e7_adaptermanager_inputoutput.py).

First, we will define our adapter manager.
In this example, we're going to cheat a little bit and combine our adapter manager (graph time) and our adapter manager impl (run time).

```python
class MyAdapterManager(AdapterManagerImpl):
    '''
    This example adapter will generate random `MyData` structs every `interval`. This simulates an upstream
    data feed, which we "connect" to only a single time. We then multiplex the results to an arbitrary
    number of subscribers via the `subscribe` method.

    We can also receive messages via the `publish` method from an arbitrary number of publishers. These messages
    are demultiplexex into a number of outputs, simulating sharing a connection to a downstream feed or responses
    to the upstream feed.
    '''
    def __init__(self, interval: timedelta):
        self._interval = interval
        self._counter = 0
        self._subscriptions = {}
        self._publications = {}
        self._running = False
        self._thread = None

    def subscribe(self, symbol):
        '''This method creates a new input adapter implementation via the manager.'''
        return _my_input_adapter(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING)

    def publish(self, data: ts['T'], symbol: str):
        '''This method creates a new output adapter implementation via the manager.'''
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
        '''This method runs in a background thread and generates random input events to push to the corresponding adapter'''
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
```

This adapter manager is a bit of a silly example, but it demonstrates the core concepts.
The adapter manager will demultiplex a shared stream (in this case, the stream defined in `_run`  is a random sequence of `MyData` structs) between all the input adapters it manages.
The input adapter itself will do nothing more than let the adapter manager know that it exists:

```python
class MyInputAdapterImpl(PushInputAdapter):
    '''Our input adapter is a very simple implementation, and just
    defers its work back to the manager who is expected to deal with
    sharing a single connection.
    '''
    def __init__(self, manager, symbol):
        manager.register_subscription(symbol, self)
        super().__init__()
```

Similarly, the adapter manager will multiplex the output adapter streams, in this case combining them into streams of print statements.
And similar to the input adapter, the output adapter does relatively little more than letting the adapter manager know that it has work available, using its triggered `on_tick` method to call the adapter manager's `_on_tick` method.

```
class MyOutputAdapterImpl(OutputAdapter):
    '''Similarly, our output adapter is simple as well, deferring
    its functionality to the manager
    '''
    def __init__(self, manager, symbol):
        manager.register_publication(symbol)
        self._manager = manager
        self._symbol = symbol
        super().__init__()

    def on_tick(self, time, value):
        self._manager._on_tick(self._symbol, value)
```

As a last step, we need to ensure that the runtime adapter implementations are registered with our graph:

```python
_my_input_adapter = py_push_adapter_def(name='MyInputAdapter', adapterimpl=MyInputAdapterImpl, out_type=ts[MyData], manager_type=MyAdapterManager, symbol=str)
_my_output_adapter = py_output_adapter_def(name='MyOutputAdapter', adapterimpl=MyOutputAdapterImpl, manager_type=MyAdapterManager, input=ts['T'], symbol=str)
```

To test this example, we will:

- instantiate our manager
- subscribe to a certain number of input adapter "streams" (which the adapter manager will demultiplex out of a single random node)
- print the data
- sink each stream into a smaller number of output adapters (which the adapter manager will multiplex into print statements)

```python
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
```

Here is the result of a single run:

```
2023-02-15 19:14:53.859951 data_1:MyData(symbol=data_1, value=0)
publication_data_1:MyData(symbol=data_1, value=0)
2023-02-15 19:14:54.610281 data_3:MyData(symbol=data_3, value=1)
publication_data_3:MyData(symbol=data_3, value=1)
2023-02-15 19:14:55.361157 data_3:MyData(symbol=data_3, value=2)
publication_data_3:MyData(symbol=data_3, value=2)
2023-02-15 19:14:56.112030 data_2:MyData(symbol=data_2, value=3)
publication_data_1:MyData(symbol=data_2, value=3)
2023-02-15 19:14:56.862881 data_2:MyData(symbol=data_2, value=4)
publication_data_1:MyData(symbol=data_2, value=4)
2023-02-15 19:14:57.613775 data_1:MyData(symbol=data_1, value=5)
publication_data_1:MyData(symbol=data_1, value=5)
2023-02-15 19:14:58.364408 data_3:MyData(symbol=data_3, value=6)
publication_data_3:MyData(symbol=data_3, value=6)
2023-02-15 19:14:59.115290 data_2:MyData(symbol=data_2, value=7)
publication_data_1:MyData(symbol=data_2, value=7)
2023-02-15 19:14:59.866160 data_2:MyData(symbol=data_2, value=8)
publication_data_1:MyData(symbol=data_2, value=8)
2023-02-15 19:15:00.617068 data_1:MyData(symbol=data_1, value=9)
publication_data_1:MyData(symbol=data_1, value=9)
2023-02-15 19:15:01.367955 data_2:MyData(symbol=data_2, value=10)
publication_data_1:MyData(symbol=data_2, value=10)
2023-02-15 19:15:02.118259 data_3:MyData(symbol=data_3, value=11)
publication_data_3:MyData(symbol=data_3, value=11)
2023-02-15 19:15:02.869170 data_2:MyData(symbol=data_2, value=12)
publication_data_1:MyData(symbol=data_2, value=12)
2023-02-15 19:15:03.620047 data_1:MyData(symbol=data_1, value=13)
publication_data_1:MyData(symbol=data_1, value=13)
closing asset publication_data_1
closing asset publication_data_3
```

Although simple, this examples demonstrates the utility of the adapters and adapter managers.
An input resource is managed by one entity, distributed across a variety of downstream subscribers.
Then a collection of streams is piped back into a single entity.
