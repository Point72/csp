## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [PushInputAdapter - Python](#pushinputadapter---python)
- [GenericPushAdapter](#genericpushadapter)
- [Realtime AdapterManager](#realtime-adaptermanager)
  - [AdapterManager - **graph-- time**](#adaptermanager---graph---time)
  - [AdapterManager - **impl-- runtime**](#adaptermanager---impl---runtime)
  - [PushInputAdapter - **--impl-- runtime**](#pushinputadapter-----impl---runtime)
  - [PushInputAdapter - **--graph-- time**](#pushinputadapter----graph---time)
  - [Example](#example)

## Introduction

There are two main categories of writing input adapters, historical and realtime.

When writing realtime adapters, you will need to implement a "push" adapter, which will get data from a separate thread that drives external events and "pushes" them into the engine as they occur.

When writing input adapters it is also very important to denote the difference between "graph building time" and "runtime" versions of your adapter.
For example, `csp.adapters.csv` has a `CSVReader` class that is used at graph building time.
**Graph build time components** solely *describe* the adapter.
They are meant to do little else than keep track of the type of adapter and its parameters, which will then be used to construct the actual adapter implementation when the engine is constructed from the graph description.
It is the runtime implementation that actual runs during the engine execution phase to process data.

For clarity of this distinction, in the descriptions below we will denote graph build time components with *--graph--* and runtime implementations with *--impl--*.

## PushInputAdapter - Python

To write a Python based `PushInputAdapter` one must write a class that derives from `csp.impl.pushadapter.PushInputAdapter`.
The derived type should the define two methods:

- `def start(self, start_time, end_time)`: this will be called at the start of the engine with the start/end times of the engine.
  start_time and end_time will be tz-unaware datetime objects in UTC time (generally these aren't needed for realtime adapters).
  At this point the adapter should open its resource / connect the data source / start any driver threads that are needed.
- `def stop(self)`: This method well be called when the engine is done running.
  At this point any open threads should be stopped and resources cleaned up.

The `PushInputAdapter` that you define will be used as the runtime *--impl–-*.
You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a `csp.impl.wiring.py_push_adapter_def`.
The `py_push_adapter_def` creates a *--graph--* time representation of your adapter:

**def py_push_adapter_def(name, adapterimpl, out_type, \*\*kwargs)**

- **`name`**: string name for the adapter
- **`adapterimpl`**: a derived implementation of
  `csp.impl.pushadapter.PushInputAdapter`
- **`out_type`**: the type of the output, should be a `ts[]` type.
  Note this can use tvar types if a subsequent argument defines the
  tvar.
- **`kwargs`**: \*\*kwargs here be passed through as arguments to the
  PushInputAdapter implementation

Note that the \*\*kwargs passed to `py_push_adapter_def` should be the names and types of the variables, like `arg1=type1, arg2=type2`.
These are the names of the kwargs that the returned input adapter will take and pass through to the `PushInputAdapter` implementation, and the types expected for the values of those args.

Example [e4_pushinput.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e4_pushinput.py) demonstrates a simple example of this.

```python
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
import csp
from csp import ts
from datetime import datetime, timedelta
import threading
import time


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time!
class MyPushAdapterImpl(PushInputAdapter):
    def __init__(self, interval):
        print("MyPushAdapterImpl::__init__")
        self._interval = interval
        self._thread = None
        self._running = False

    def start(self, starttime, endtime):
        """ start will get called at the start of the engine, at which point the push
        input adapter should start its thread that will push the data onto the adapter. Note
        that push adapters will ALWAYS have a separate thread driving ticks into the csp engine thread
        """
        print("MyPushAdapterImpl::start")
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """ stop will get called at the end of the run, at which point resources should
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
MyPushAdapter = py_push_adapter_def('MyPushAdapter', MyPushAdapterImpl, ts[int], interval=timedelta)
```

Note how line 41 calls **self.push_tick**.
This is the call to get data from the adapter thread ticking into the CSP engine.

Now `MyPushAdapter` can be called in graph code to create a timeseries that is sourced by `MyPushAdapterImpl`:

```python
@csp.graph
def my_graph():
    # At this point we create the graph-time representation of the input adapter. This will be converted
    # into the impl once the graph is done constructing and the engine is created in order to run
    data = MyPushAdapter(timedelta(seconds=1))
    csp.print('data', data)
```

## GenericPushAdapter

If you dont need as much control as `PushInputAdapter` provides, or if you have some existing source of data on a thread you can't control, another option is to use the higher-level abstraction `csp.GenericPushAdapter`.
`csp.GenericPushAdapter` wraps a `csp.PushInputAdapter` implementation internally and provides a simplified interface.
The downside of `csp.GenericPushAdapter` is that you lose some control of when the input feed starts and stop.

Lets take a look at the example found in [e1_generic_push_adapter.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e1_generic_push_adapter.py):

```python
# This is an example of some separate thread providing data
class Driver:
    def __init__(self, adapter : csp.GenericPushAdapter):
        self._adapter = adapter
        self._active = False
        self._thread = None

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._active:
            self._active = False
            self._thread.join()

    def _run(self):
        print("driver thread started")
        counter = 0
        # Optionally, we can wait for the adapter to start before proceeding
        # Alternatively we can start pushing data, but push_tick may fail and return False if
        # the csp engine isn't ready yet
        self._adapter.wait_for_start()

        while self._active and not self._adapter.stopped():
            self._adapter.push_tick(counter)
            counter += 1
            time.sleep(1)

@csp.graph
def my_graph():
    adapter = csp.GenericPushAdapter(int)
    driver = Driver(adapter)
    # Note that the driver thread starts *before* the engine is started here, which means some ticks may potentially get dropped if the
    # data source doesn't wait for the adapter to start. This may be ok for some feeds, but not others
    driver.start()

    # Lets be nice and shutdown the driver thread when the engine is done
    csp.schedule_on_engine_stop(driver.stop)
```

In this example we have this dummy `Driver` class which simply represents some external source of data which arrives on a thread that's completely independent of the engine.
We pass along a `csp.GenericInputAdapter` instance to this thread, which can then call adapter.push_tick to get data into the engine (see line 27).

On line 24 we can also see an optional feature which allows the unrelated thread to wait for the adapter to be ready to accept data before ticking data onto it.
If push_tick is called before the engine starts / the adapter is ready to receive data, it will simply drop the data.
Note that GenericPushAadapter.push_tick will return a bool to indicate whether the data was successfully pushed to the engine or not.

## Realtime `AdapterManager`

In most cases you will likely want to expose a single source of data into multiple input adapters.
For this use case your adapter should define an `AdapterManager` *--graph--* time component, and `AdapterManagerImpl` *--impl--* runtime component.
The `AdapterManager` *--graph--* time component just represents the parameters needed to create the *--impl--* `AdapterManager`.
Its the *--impl--* that will have the actual implementation that will open the data source, parse the data and provide it to individual Adapters.

Similarly you will need to define a derived `PushInputAdapter` *--impl--* component to handle events directed at an individual time series adapter.

**NOTE** It is highly recommended not to open any resources in the *--graph--* time component.
Graph time components can be pruned and/or memoized into a single instance, opening resources at graph time shouldn't be necessary.

### AdapterManager - **graph-- time**

The graph time `AdapterManager` doesn't need to derive from any interface.
It should be initialized with any information the impl needs in order to open/process the data source (ie activemq connection information, server host port, multicast channels, config files, etc etc).
It should also have an API to create individual timeseries adapters.
These adapters will then get passed the adapter manager *--impl--* as an argument when they are created, so that they can register themselves for processing.
The `AdapterManager` also needs to define a **\_create** method.
The **\_create** is the bridge between the *--graph--* time `AdapterManager` representation and the runtime *--impl--* object.
**\_create** will be called on the *--graph--* time `AdapterManager` which will in turn create the *--impl--* instance.
\_create will get two arguments, engine (this represents the runtime engine object that will run the graph) and  memo dict which can optionally be used for any memoization that on might want.

Lets take a look at the example found in [e5_adaptermanager_pushinput.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e5_adaptermanager_pushinput.py):

```python
# This object represents our AdapterManager at graph time. It describes the manager's properties
# and will be used to create the actual impl when its time to build the engine
class MyAdapterManager:
    def __init__(self, interval: timedelta):
        """
        Normally one would pass properties of the manager here, ie filename,
        message bus, etc
        """
        self._interval = interval

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        """ User facing API to subscribe to a timeseries stream from this adapter manager """
        # This will return a graph-time timeseries edge representing and edge from this
        # adapter manager for the given symbol / arguments
        return MyPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        """ This method will get called at engine build time, at which point the graph time manager representation
        will create the actual impl that will be used for runtime
        """
        # Normally you would pass the arguments down into the impl here
        return MyAdapterManagerImpl(engine, self._interval)
```

- **\_\_init\_\_** - as you can see, all \_\_init\_\_ does is keep the parameters that the impl will need.
- **subscribe** - API to create an individual timeseries / edge from this file for the given symbol.
  The interface defined here is up to the adapter writer, but generally "subscribe" is recommended, and it should take any number of arguments needed to define a single stream of data.
  *MyPushAdapter* is the *--graph--* time representation of the edge, which will be described below.
  We pass it *self* as its first argument, which will be used to create the `AdapterManager` *--impl--*
- **\_create** - the method to create the *--impl--* object from the given *--graph--* time representation of the manager

`MyAdapterManager` would then be used in graph building code like so:

```python
adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
data = adapter_manager.subscribe('AAPL', push_mode=csp.PushMode.LAST_VALUE)
csp.print(symbol + " last_value", data)
```

### AdapterManager - **impl-- runtime**

The `AdapterManager` *--impl--* is responsible for opening the data source, parsing and processing all the data and managing all the adapters it needs to feed.
The impl class should derive from `csp.impl.adaptermanager.AdapterManagerImpl` and implement the following methods:

- **start(self,starttime,endtime)**: this is called when the engine starts up.
  At this point the impl should open the resource providing the data and start up any thread(s) needed to listen to and react to external data.
  starttime/endtime will be tz-unaware datetime objects in UTC time, though typically these aren't needed for realtime adapters
- **`stop(self)`**: this is called at the end of the run, resources should be cleaned up at this point
- **`process_next_sim_timeslice(self, now)`**: this is used by sim adapters, for realtime adapter managers we simply return None

In the example manager, we spawn a processing thread in the `start()` call.
This thread runs in a loop until it is shutdown, and will generate random data to tick out to the registered input adapters.
Data is passed to a given adapter by calling `push_tick()`.

### PushInputAdapter - **--impl-- runtime**

Users will need to define `PushInputAdapter` derived types to represent the individual timeseries adapter *--impl--* objects.
Objects should derive from `csp.impl.pushadapter.PushInputAdapter`.

`PushInputAdapter` defines a method `push_tick()` which takes the value to feed the input timeseries.

### PushInputAdapter - **--graph-- time**

Similar to the stand alone `PushInputAdapter` described above, we need to define a graph-time construct that represents a `PushInputAdapter` edge.
In order to define this we use `py_push_adapter_def` again, but this time we pass the adapter manager *--graph--* time type so that it gets constructed properly.
When the `PushInputAdapter` instance is created it will also receive an instance of the adapter manager *--impl–-*, which it can then self-register on.

```python
def py_push_adapter_def (name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs):
"""
Create a graph representation of a python push input adapter.
:param name: string name for the adapter
:param adapterimpl: a derived implementation of csp.impl.pushadapter.PushInputAdapter
:param out_type: the type of the output, should be a ts[] type. Note this can use tvar types if a subsequent argument defines the tvar
:param manager_type: the type of the graph time representation of the AdapterManager that will manage this adapter
:param kwargs: **kwargs will be passed through as arguments to the ManagedSimInputAdapter implementation
the first argument to the implementation will be the adapter manager impl instance
"""
```

### Example

Continuing with the *--graph--* time `AdapterManager` described above, we
now define the impl:

```python
# This is the actual manager impl that will be created and executed during runtime
class MyAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, interval):
        super().__init__(engine)

        # These are just used to simulate a data source
        self._interval = interval
        self._counter = 0

        # We will keep track of requested input adapters here
        self._inputs = {}

        # Our driving thread, all realtime adapters will need a separate thread of execution that
        # drives data into the engine thread
        self._running = False
        self._thread = None

    def start(self, starttime, endtime):
        """ start will get called at the start of the engine run. At this point
            one would start up the realtime data source / spawn the driving thread(s) and
            subscribe to the needed data """
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        """ This will be called at the end of the engine run, at which point resources should be
            closed and cleaned up """
        if self._running:
            self._running = False
            self._thread.join()

    def register_input_adapter(self, symbol, adapter):
        """ Actual PushInputAdapters will self register when they are created as part of the engine
            This is the place we gather all requested input adapters and their properties
        """
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        # Keep a list of adapters by key in case we get duplicate adapters (should be memoized in reality)
        self._inputs[symbol].append(adapter)

    def process_next_sim_timeslice(self, now):
        """ This method is only used by simulated / historical adapters, for realtime we just return None """
        return None

    def _run(self):
        """ Our driving thread, in reality this will be reacting to external events, parsing the data and
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
```

Then we define our `PushInputAdapter` *--impl--*, which basically just
self-registers with the adapter manager *--impl--* upon construction. We
also define our `PushInputAdapter` *--graph--* time construct using `py_push_adapter_def`.

```python
# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time. a managed sim adapter impl will get the
# adapter manager runtime impl as its first argument
class MyPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        print(f"MyPushAdapterImpl::__init__ {symbol}")
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


MyPushAdapter = py_push_adapter_def('MyPushAdapter', MyPushAdapterImpl, ts[MyData], MyAdapterManager, symbol=str)
```

And then we can run our adapter in a CSP graph

```python
@csp.graph
def my_graph():
    print("Start of graph building")

    adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
    symbols = ['AAPL', 'IBM', 'TSLA', 'GS', 'JPM']
    for symbol in symbols:
        # your data source might tick faster than the engine thread can consume it
        # push_mode can be used to buffered up tick events will get processed
        # LAST_VALUE will conflate and only tick the latest value since the last cycle
        data = adapter_manager.subscribe(symbol, csp.PushMode.LAST_VALUE)
        csp.print(symbol + " last_value", data)

        # BURST will change the timeseries type from ts[T] to ts[[T]] (list of ticks)
        # that will tick with all values that have buffered since the last engine cycle
        data = adapter_manager.subscribe(symbol, csp.PushMode.BURST)
        csp.print(symbol + " burst", data)

        # NON_COLLAPSING will tick all events without collapsing, unrolling the events
        # over multiple engine cycles
        data = adapter_manager.subscribe(symbol, csp.PushMode.NON_COLLAPSING)
        csp.print(symbol + " non_collapsing", data)

    print("End of graph building")


csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
```

Do note that realtime adapters will only run in realtime engines (note the `realtime=True` argument to `csp.run`).

## Engine shutdown

In case a pushing thread hits a terminal error, an exception can be passed to the main engine thread to shut down gracefully through a `shutdown_engine(exc: Exception)` method exposed by `PushInputAdapter`, `PushPullInputAdapter` and `AdapterManagerImpl`.

For example:

```python
def _run(self):
   while self._running:
        try:
            requests.get(endpoint) # API call over a network, may fail
        except Exception as exc:
            self.shutdown_engine(exc)
```
