## Input Adapters

There are two main categories of writing input adapters, historical and realtime.
When writing historical adapters you will need to implement a "pull" adapter, which pulls data from a historical data source in time order, one event at a time.
There are also ManagedSimAdapters for feeding multiple "managed" pull adapters from a single source (more on that below).
When writing realtime adapters, you will need to implement a "push" adapter, which will get data from a separate thread that drives external events and "pushes" them into the engine as they occur.

When writing input adapters it is also very important to denote the difference between "graph building time" and "runtime" versions of your adapter.
For example, `csp.adapters.csv` has a `CSVReader` class that is used at graph building time.
**Graph build time components** solely *describe* the adapter.
They are meant to do little else than keep track of the type of adapter and its parameters, which will then be used to construct the actual adapter implementation when the engine is constructed from the graph description.
It is the runtime implementation that actual runs during the engine execution phase to process data.

For clarity of this distinction, in the descriptions below we will denote graph build time components with *--graph--* and runtime implementations with *--impl--*.

### Historical Adapters

There are two flavors of historical input adapters that can be written.
The simplest one is a PullInputAdapter.
A PullInputAdapter can be used to convert a single source into a single timeseries.
The csp.curve implementation is a good example of this.
Single source to single timeseries adapters are of limited use however, and the more typical use case is for AdapterManager based input adapters to service multiple InputAdapters from a single source.
For this one would use an AdapterManager to coordinate processing of the data source, and ManagedSimInputAdapter as the individual timeseries providers.

#### PullInputAdapter - Python

To write a Python based PullInputAdapter one must write a class that derives from csp.impl.pulladapter.PullInputAdapter.
The derived type should the define two methods:

- `def start(self, start_time, end_time)`: this will be called at the start of the engine with the start/end times of the engine.
  start_Time and end_time will be tz-unaware datetime objects in UTC time.
  At this point the adapter should open its resource and seek to the requested starttime.
- `def next(self)`: this method will be repeatedly called by the engine.
  The adapter should return the next event as a time,value tuple.
  If there are no more events, then the method should return None

The PullInputAdapter that you define will be used as the runtime *--impl–-*.
You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a csp.impl.wiring.py_pull_adapter_def.
The py_pull_adapter_def creates a *--graph--* time representation of your adapter:

```python
def py_pull_adapter_def(name, adapterimpl, out_type, **kwargs)
```

- **`name`**: string name for the adapter
- **`adapterimpl`**: a derived implementation of csp.impl.pulladapter.PullInputAdapter
- **`out_type`**: the type of the output, should be a `ts[]` type. Note this can use tvar types if a subsequent argument defines the tvar
- **`kwargs`**: \*\*kwargs here be passed through as arguments to the PullInputAdapter implementation

Note that the \*\*kwargs passed to py_pull_adapter_def should be the names and types of the variables, like arg1=type1, arg2=type2.
These are the names of the kwargs that the returned input adapter will take and pass through to the PullInputAdapter implementation, and the types expected for the values of those args.

csp.curve is a good simple example of this:

```python
import copy
from csp.impl.pulladapter import PullInputAdapter
from csp.impl.wiring import py_pull_adapter_def
from csp import ts
from datetime import timedelta


class Curve(PullInputAdapter):
    def __init__(self, typ, data):
        ''' data should be a list of tuples of (datetime, value) or (timedelta, value)'''
        self._data = data
        self._index = 0
        super().__init__()

    def start(self, start_time, end_time):
        if isinstance(self._data[0][0], timedelta):
            self._data = copy.copy(self._data)
            for idx, data in enumerate(self._data):
                self._data[idx] = (start_time + data[0], data[1])

        while self._index < len(self._data) and self._data[self._index][0] < start_time:
            self._index += 1

        super().start(start_time, end_time)

    def next(self):

        if self._index < len(self._data):
            time, value = self._data[self._index]
            if time <= self._end_time:
                self._index += 1
                return time, value
        return None


curve = py_pull_adapter_def('curve', Curve, ts['T'], typ='T', data=list)
```

Now curve can be called in graph code to create a curve input adapter:

```python
x = csp.curve(int, [ (t1, v1), (t2, v2), .. ])
csp.print('x', x)
```

See example "e_14_user_adapters_01_pullinput.py for

#### PullInputAdapter - C++

**Step 1)** PullInputAdapter impl

Similar to the Python PullInputAdapter API is the c++ API which one can leverage to improve performance of an adapter implementation.
The *--impl--* is very similar to python pull adapter.
One should derive from `PullInputAdapter<T>`, a templatized base class (templatized on the type of the timeseries) and define these methods:

- **`start(DateTime start, DateTime end)`**: similar to python API start, called when engine starts.
  Open resource and seek to start time here
- **`stop()`**: called on engine shutdown, cleanup resource
- **`bool next(DateTime & t, T & value)`**: if there is data to provide, sets the next time and value for the adapter and returns true.
  Otherwise, return false

**Step 2)** Expose creator func to python

Now that we have a c++ impl defined, we need to expose a python creator for it.
Define a method that conforms to the signature

```cpp
csp::InputAdapter * create_my_adapter(
    csp::AdapterManager * manager,
    PyEngine * pyengine,
    PyTypeObject * pyType,
    PushMode pushMode,
    PyObject * args)
```

- **`manager`**: will be nullptr for pull adapters
- **`pyengine `**: PyEngine engine wrapper object
- **`pyType`**: this is the type of the timeseries input adapter to be created as a PyTypeObject.
  one can switch on this type using switchPyType to create the properly typed instance
- **`pushMode`**: the csp PushMode for the adapter (pass through to base InputAdapter)
- **`args`**: arguments to pass to the adapter impl

Then simply register the creator method:

**`REGISTER_INPUT_ADAPTER(_my_adapter, create_my_adapter)`**

This will register methodname onto your python module, to be accessed as your module.methodname.
Note this uses csp/python/InitHelpers which is used in the \_cspimpl module.
To do this in a separate python module, you need to register InitHelpers in that module.

**Step 3)** Define your *--graph–-* time adapter

One liner now to wrap your impl in a graph time construct using csp.impl.wiring.input_adapter_def:

```python
my_adapter = input_adapter_def('my_adapter', my_module._my_adapter, ts[int], arg1=int, arg2={str:'foo'})
```

my_adapter can now be called with arg1, arg2 to create adapters in your graph.
Note that the arguments are typed using v=t syntax.  v=(t,default) is used to define arguments with defaults.

Also note that all input adapters implicitly get a push_mode argument that is defaulted to csp.PushMode.LAST_VALUE.

#### ManagedSimInputAdapter - Python

In most cases you will likely want to expose a single source of data into multiple input adapters.
For this use case your adapter should define an AdapterManager *--graph--* time component, and AdapterManagerImpl *--impl--* runtime component.
The AdapterManager *--graph--* time component just represents the parameters needed to create the *--impl--* AdapterManager.
Its the *--impl--* that will have the actual implementation that will open the data source, parse the data and provide it to individual Adapters.

Similarly you will need to define a derived ManagedSimInputAdapter *--impl--* component to handle events directed at an individual time series adapter.

**NOTE** It is highly recommended not to open any resources in the *--graph--* time component.
graph time components can be pruned and/or memoized into a single instance, opening resources at graph time shouldn't be necessary.

#### AdapterManager - **--graph-- time**

The graph time AdapterManager doesn't need to derive from any interface.
It should be initialized with any information the impl needs in order to open/process the data source (ie csv file, time column, db connection information, etc etc).
It should also have an API to create individual timeseries adapters.
These adapters will then get passed the adapter manager *--impl--* as an argument where they are created, so that they can register themselves for processing.
The AdapterManager also needs to define a **\_create** method.
The **\_create** is the bridge between the *--graph--* time AdapterManager representation and the runtime *--impl--* object.
**\_create** will be called on the *--graph--* time AdapterManager which will in turn create the *--impl--* instance.
\_create will get two arguments, engine (this represents the runtime engine object that will run the graph) and a memo dict which can optionally be used for any memoization that on might want.

Lets take a look at CSVReader as an example:

```python
# GRAPH TIME
class CSVReader:
    def __init__(self, filename, time_converter, delimiter=',', symbol_column=None):
        self._filename = filename
        self._symbol_column = symbol_column
        self._delimiter = delimiter
        self._time_converter = time_converter

    def subscribe(self, symbol, typ, field_map=None):
        return CSVReadAdapter(self, symbol, typ, field_map)

    def _create(self, engine, memo):
        return CSVReaderImpl(engine, self)
```

- **`__init__`**: as you can see, all `__init__` does is keep the parameters that the impl will need.
- **`subscribe`**: API to create an individual timeseries / edge from this file for the given symbol.
  typ denotes the type of the timeseries to create (ie `ts[int]`) and field_map is used for mapping columns onto csp.Struct types.
  Note that subscribe returns a CSVReadAdapter instance.
  CSVReadAdapter is the *--graph--* time representation of the edge (similar to how we defined csp.curve above).
  We pass it `self` as its first argument, which will be used to create the AdapterManager *--impl--*
- **`\_create`**: the method to create the *--impl--* object from the given *--graph--* time representation of the manager

The CSVReader would then be used in graph building code like so:

```python
reader = CSVReader('my_data.csv', time_formatter, symbol_column='SYMBOL', delimiter='|')
# aapl will represent a ts[PriceQuantity] edge that will tick with rows from
# the csv file matching on SYMBOL column AAPL
aapl = reader.subscribe('AAPL', PriceQuantity)
```

##### AdapterManager - **--impl-- runtime**

The AdapterManager *--impl--* is responsible for opening the data source, parsing and processing through all the data and managing all the adapters it needs to feed.
The impl class should derive from csp.impl.adaptermanager.AdapterManagerImpl and implement the following methods:

- **`start(self,starttime,endtime)`**: this is called when the engine starts up.
  At this point the impl should open the resource providing the data and seek to starttime.
  starttime/endtime will be tz-unaware datetime objects in UTC time
- **`stop(self)`**: this is called at the end of the run, resources should be cleaned up at this point
- **`process_next_sim_timeslice(self, now)`**: this method will be called multiple times through the run.
  The initial call will provide now with starttime.
  The impl's responsibility is to process all data at the given timestamp (more on how to do this below).
  The method should return the next time in the data source, or None if there is no more data to process.
  The method will be called again with the provided timestamp as "now" in the next iteration.
  **NOTE** that process_next_sim_timeslice is required to move ahead in time.
  In most cases the resource data can be supplied in time order, if not it would have to be sorted up front.

process_next_sim_timeslice should parse data for a given time/row of data and then push it through to any registered ManagedSimInputAdapter that matches on the given row

##### ManagedSimInputAdapter - **--impl-- runtime**

Users will need to define ManagedSimInputAdapter derived types to represent the individual timeseries adapter *--impl--* objects.
Objects should derive from csp.impl.adaptermanager.ManagedSimInputAdapter.

ManagedSimInputAdapter.`__init__` takes two arguments:

- **`typ`**: this is the type of the timeseries, ie int for a `ts[int]`
- **`field_map`**: Optional, field_map is a dictionary used to map source column names → csp.Struct field names.

ManagedSimInputAdapter defines a method `push_tick()` which takes the value to feed the input for given timeslice (as defined by "now" at the adapter manager level).
There is also a convenience method called `process_dict()` which will take a dictionary of `{column : value}` entries and convert it properly into the right value based on the given **field_map.**

##### \*\*ManagedSimInputAdapter - **--graph-- time**

As with the csp.curve example, we need to define a graph-time construct that represents a ManagedSimInputAdapter edge.
In order to define this we use py_managed_adapter_def.
py_managed_adapter_defis AdapterManager "aware" and will properly create the AdapterManager *--impl--* the first time its encountered.
It will then pass the manager impl as an argument to the ManagedSimInputAdapter.

```python
def py_managed_adapter_def(name, adapterimpl, out_type, manager_type, **kwargs):
"""
Create a graph representation of a python managed sim input adapter.
:param name: string name for the adapter
:param adapterimpl: a derived implementation of csp.impl.adaptermanager.ManagedSimInputAdapter
:param out_type: the type of the output, should be a ts[] type. Note this can use tvar types if a subsequent argument defines the tvar
:param manager_type: the type of the graph time representation of the AdapterManager that will manage this adapter
:param kwargs: **kwargs will be passed through as arguments to the ManagedSimInputAdapter implementation
the first argument to the implementation will be the adapter manager impl instance
"""
```

##### Example - CSVReader

Putting this all together lets take a look at a CSVReader implementation
and step through what's going on:

```python
import csv as pycsv
from datetime import datetime

from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
from csp.impl.wiring import pymanagedadapterdef

# GRAPH TIME
class CSVReader:
    def __init__(self, filename, time_converter, delimiter=',', symbol_column=None):
        self._filename = filename
        self._symbol_column = symbol_column
        self._delimiter = delimiter
        self._time_converter = time_converter

    def subscribe(self, symbol, typ, field_map=None):
        return CSVReadAdapter(self, symbol, typ, field_map)

    def _create(self, engine, memo):
        return CSVReaderImpl(engine, self)
```

Here we define CSVReader, our AdapterManager *--graph--* time representation.
It holds the parameters that will be used for the impl, it implements a `subscribe()` call for users to create timeseries and defines a \_create method to create a runtime *--impl–-* instance from the graphtime representation.
Note how on line 17 we pass self to the CSVReadAdapter, this is what binds the input adapter to this AdapterManager

```python
# RUN TIME
class CSVReaderImpl(AdapterManagerImpl):                     # 1
    def __init__(self, engine, adapterRep):                  # 2
        super().__init__(engine)                             # 3
                                                             # 4
        self._rep = adapterRep                               # 5
        self._inputs = {}                                    # 6
        self._csv_reader = None                              # 7
        self._next_row = None                                # 8
                                                             # 9
    def start(self, starttime, endtime):                     # 10
        self._csv_reader = pycsv.DictReader(                 # 11
            open(self._rep._filename, 'r'),                  # 12
            delimiter=self._rep._delimiter                   # 13
        )                                                    # 14
        self._next_row = None                                # 15
                                                             # 16
        for row in self._csv_reader:                         # 17
            time = self._rep._time_converter(row)            # 18
            self._next_row = row                             # 19
            if time >= starttime:                            # 20
                break                                        # 21
                                                             # 22
    def stop(self):                                          # 23
        self._csv_reader = None                              # 24
                                                             # 25
    def register_input_adapter(self, symbol, adapter):       # 26
        if symbol not in self._inputs:                       # 27
            self._inputs[symbol] = []                        # 28
        self._inputs[symbol].append(adapter)                 # 29
                                                             # 30
    def process_next_sim_timeslice(self, now):               # 31
        if not self._next_row:                               # 32
            return None                                      # 33
                                                             # 34
        while True:                                          # 35
            time = self._rep._time_converter(self._next_row) # 36
            if time > now:                                   # 37
                return time                                  # 38
            self.process_row(self._next_row)                 # 39
            try:                                             # 40
                self._next_row = next(self._csv_reader)      # 41
            except StopIteration:                            # 42
                return None                                  # 43
                                                             # 44
    def process_row(self, row):                              # 45
        symbol = row[self._rep._symbol_column]               # 46
        if symbol in self._inputs:                           # 47
            for input in self._inputs.get(symbol, []):       # 48
                input.process_dict(row)                      # 49
```

CSVReaderImpl is the runtime *--impl–-*.
It gets created when the engine is being built from the described graph.

- **lines 10-21 - start()**: this is the start method that gets called with the time range the graph will be run against.
  Here we open our resource (pycsv.DictReader) and scan t through the data until we reach the requested starttime.

- **lines 23-24 - stop()**: this is the stop call that gets called when the engine is done running and is shutdown, we free our resource here

- **lines 26-29**: the CSVReader allows one to subscribe to many symbols from one file.
  symbols are keyed by a provided SYMBOL column.
  The individual adapters will self-register with the CSVReaderImpl when they are created with the requested symbol.
  CSVReaderImpl keeps track of what adapters have been registered for what symbol in its self.\_inputs map

- **lines 31-43**: this is main method that gets invoked repeatedly throughout the run.
  For every distinct timestamp in the file, this method will get invoked once and the method is expected to go through the resource data for all points with time now, process the row and push the data to any matching adapters.
  The method returns the next timestamp when its done processing all data for "now", or None if there is no more data.
  **NOTE** that the csv impl expects the data to be in time order.
  process_next_sim_timeslice must advance time forward.

- **lines 45-49**: this method takes a row of data (provided as a dict from DictReader), extracts the symbol and pushes the row through to all input adapters that match

```python
class CSVReadAdapterImpl(ManagedSimInputAdapter):            # 1
    def __init__(self, managerImpl, symbol, typ, field_map): # 2
        managerImpl.register_input_adapter(symbol, self)     # 3
        super().__init__(typ, field_map)                     # 4
                                                             # 5
CSVReadAdapter = py_managed_adapter_def(                     # 6
    'csvadapter',
    CSVReadAdapterImpl,
    ts['T'],
    CSVReader,
    symbol=str,
    typ='T',
    fieldMap=(object, None)
)
```

- **line 3**: this is where the instance of an adapter *--impl--* registers itself with the CSVReaderImpl.
- **line 6+**: this is where we define CSVReadAdapter, the *--graph--* time representation of a CSV adapter, returned from CSVReader.subscribe

See example "e_14_user_adapters_02_adaptermanager_siminput" for another example of how to write a managed sim adapter manager.

### Realtime Adapters

#### PushInputAdapter - python

To write a Python based PushInputAdapter one must write a class that derives from csp.impl.pushadapter.PushInputAdapter.
The derived type should the define two methods:

- `def start(self, start_time, end_time)`: this will be called at the start of the engine with the start/end times of the engine.
  start_time and end_time will be tz-unaware datetime objects in UTC time (generally these aren't needed for realtime adapters).
  At this point the adapter should open its resource / connect the data source / start any driver threads that are needed.
- `def stop(self)`: This method well be called when the engine is done running.
  At this point any open threads should be stopped and resources cleaned up.

The PushInputAdapter that you define will be used as the runtime *--impl–-*.
You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a csp.impl.wiring.py_push_adapter_def.
The py_push_adapter_def creates a *--graph--* time representation of your adapter:

**def py_push_adapter_def(name, adapterimpl, out_type, \*\*kwargs)**

- **`name`**: string name for the adapter
- **`adapterimpl`**: a derived implementation of
  csp.impl.pushadapter.PushInputAdapter
- **`out_type`**: the type of the output, should be a ts\[\] type.
  Note this can use tvar types if a subsequent argument defines the
  tvar
- **`kwargs`**: \*\*kwargs here be passed through as arguments to the
  PushInputAdapter implementation

Note that the \*\*kwargs passed to py_push_adapter_def should be the names and types of the variables, like arg1=type1, arg2=type2.
These are the names of the kwargs that the returned input adapter will take and pass through to the PushInputAdapter implementation, and the types expected for the values of those args.

Example e_14_user_adapters_03_pushinput.py demonstrates a simple example of this

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
This is the call to get data from the adapter thread ticking into the csp engine

Now MyPushAdapter can be called in graph code to create a timeseries that is sourced by MyPushAdapterImpl

```python
@csp.graph
def my_graph():
    # At this point we create the graph-time representation of the input adapter. This will be converted
    # into the impl once the graph is done constructing and the engine is created in order to run
    data = MyPushAdapter(timedelta(seconds=1))
    csp.print('data', data)
```

#### GenericPushAdapter

If you dont need as much control as PushInputAdapter provides, or if you have some existing source of data on a thread you can't control, another option is to use the higher-level abstraction csp.GenericPushAdapter.
csp.GenericPushAdapter wraps a csp.PushInputAdapter implementation internally and provides a simplified interface.
The downside of csp.GenericPushAdapter is that you lose some control of when the input feed starts and stop.

Lets take a look at the example found in "e_14_generic_push_adapter"

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

In this example we have this dummy Driver class which simply represents some external source of data which arrives on a thread that's completely independent of the engine.
We pass along a csp.GenericInputAdapter instance to this thread, which can then call adapter.push_tick to get data into the engine (see line 27).

On line 24 we can also see an optional feature which allows the unrelated thread to wait for the adapter to be ready to accept data before ticking data onto it.
If push_tick is called before the engine starts / the adapter is ready to receive data, it will simply drop the data.
Note that GenericPushAadapter.push_tick will return a bool to indicate whether the data was successfully pushed to the engine or not.

### Realtime AdapterManager

In most cases you will likely want to expose a single source of data into multiple input adapters.
For this use case your adapter should define an AdapterManager *--graph--* time component, and AdapterManagerImpl *--impl--* runtime component.
The AdapterManager *--graph--* time component just represents the parameters needed to create the *--impl--* AdapterManager.
Its the *--impl--* that will have the actual implementation that will open the data source, parse the data and provide it to individual Adapters.

Similarly you will need to define a derived PushInputAdapter *--impl--* component to handle events directed at an individual time series adapter.

**NOTE** It is highly recommended not to open any resources in the *--graph--* time component.
Graph time components can be pruned and/or memoized into a single instance, opening resources at graph time shouldn't be necessary.

#### AdapterManager - **graph-- time**

The graph time AdapterManager doesn't need to derive from any interface.
It should be initialized with any information the impl needs in order to open/process the data source (ie activemq connection information, server host port, multicast channels, config files, etc etc).
It should also have an API to create individual timeseries adapters.
These adapters will then get passed the adapter manager *--impl--* as an argument when they are created, so that they can register themselves for processing.
The AdapterManager also needs to define a **\_create** method.
The **\_create** is the bridge between the *--graph--* time AdapterManager representation and the runtime *--impl--* object.
**\_create** will be called on the *--graph--* time AdapterManager which will in turn create the *--impl--* instance.
\_create will get two arguments, engine (this represents the runtime engine object that will run the graph) and  memo dict which can optionally be used for any memoization that on might want.

Lets take a look at the example found in
"e_14_user_adapters_04_adaptermanager_pushinput"

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

- **\_\_init\_\_** - as you can see, all \_\_init\_\_ does is keep the parameters that the impl will need.
- **subscribe** - API to create an individual timeseries / edge from this file for the given symbol.
  The interface defined here is up to the adapter writer, but generally "subscribe" is recommended, and it should take any number of arguments needed to define a single stream of data.
  *MyPushAdapter* is the *--graph--* time representation of the edge, which will be described below.
  We pass it *self* as its first argument, which will be used to create the AdapterManager *--impl--*
- **\_create** - the method to create the *--impl--* object from the given *--graph--* time representation of the manager

MyAdapterManager would then be used in graph building code like so:

```python
adapter_manager = MyAdapterManager(timedelta(seconds=0.75))
data = adapter_manager.subscribe('AAPL', push_mode=csp.PushMode.LAST_VALUE)
csp.print(symbol + " last_value", data)
```

## AdapterManager - **impl-- runtime**

The AdapterManager *--impl--* is responsible for opening the data source, parsing and processing all the data and managing all the adapters it needs to feed.
The impl class should derive from csp.impl.adaptermanager.AdapterManagerImpl and implement the following methods:

- **start(self,starttime,endtime)**: this is called when the engine starts up.
  At this point the impl should open the resource providing the data and start up any thread(s) needed to listen to and react to external data.
  starttime/endtime will be tz-unaware datetime objects in UTC time, though typically these aren't needed for realtime adapters
- **`stop(self)`**: this is called at the end of the run, resources should be cleaned up at this point
- **`process_next_sim_timeslice(self, now)`**: this is used by sim adapters, for realtime adapter managers we simply return None

In the example manager, we spawn a processing thread in the `start()` call.
This thread runs in a loop until it is shutdown, and will generate random data to tick out to the registered input adapters.
Data is passed to a given adapter by calling **push_tick**()

#### PushInputAdapter - **--impl-- runtime**

Users will need to define PushInputAdapter derived types to represent the individual timeseries adapter *--impl--* objects.
Objects should derive from csp.impl.pushadapter.PushInputAdapter.

PushInputAdapter defines a method `push_tick()` which takes the value to feed the input timeseries.

#### PushInputAdapter - **--graph-- time**

Similar to the stand alone PushInputAdapter described above, we need to define a graph-time construct that represents a PushInputAdapter edge.
In order to define this we use py_push_adapter_def again, but this time we pass the adapter manager *--graph--* time type so that it gets constructed properly.
When the PushInputAdapter instance is created it will also receive an instance of the adapter manager *--impl–-*, which it can then self-register on/

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

#### Example

Continuing with the --graph-- time AdapterManager described above, we
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

Then we define our PushInputAdapter --impl--, which basically just
self-registers with the adapter manager --impl-- upon construction. We
also define our PushInputAdapter *--graph--* time construct using `py_push_adapter_def`.

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

And then we can run our adapter in a csp graph

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

## Output Adapters

Output adapters are used to define graph outputs, and they differ from input adapters in a number of important ways.
Output adapters also differ from terminal nodes, e.g. regular `csp.node` instances that do not define outputs, and instead consume and emit their inputs inside their `csp.ticked`  blocks.

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
Unlike input adapters, output adapters do not need to differentiate between *historical* and *realtime* mode.
Input adapters drive the execution of the graph, whereas output adapters are reactive to their input nodes and subject to the graph's execution.

However, there are a number of reasons why you might want to define an output adapter instead of using a vanilla node.
The most important of these is when you want to share resources across a number of output adapters (e.g. with a Manager), or between an input and an output node, e.g. reading data from a websocket, routing it through your csp graph, and publishing data *to the same websocket connection*.
For most use cases, a vanilla csp node will suffice, but let's explore some anyway.

### OutputAdapter - Python

To write a Python based OutputAdapter one must write a class that derives from `csp.impl.outputadapter.OutputAdapter`.
The derived type should define the method:

- `def on_tick(self, time: datetime, value: object)`: this will be called when the input to the output adapter ticks.

The OutputAdapter that you define will be used as the runtime *--impl–-*.  You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a csp.impl.wiring.py_output_adapter_def.
The py_output_adapter_def creates a *--graph--* time representation of your adapter:

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
This example is available in `e_14_user_adapters_05_adaptermanager_inputoutput` .

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
The adapter manager will demultiplex a shared stream (in this case, the stream defined in `_run`  is a random sequence of `MyData` structs) between all the input adapters it manages.
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
And similar to the input adapter, the output adapter does relatively little more than letting the adapter manager know that it has work available, using its triggered `on_tick` method to call the adapter manager's `_on_tick` method.

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
