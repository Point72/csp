## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Types of Historical Adapters](#types-of-historical-adapters)
- [PullInputAdapter](#pullinputadapter)
  - [PullInputAdapter - Python](#pullinputadapter---python)
  - [PullInputAdapter - C++](#pullinputadapter---c)
- [AdapterManager and ManagedSimInputAdapter - Python](#adaptermanager-and-managedsiminputadapter---python)
  - [AdapterManager - **--graph-- time**](#adaptermanager-----graph---time)
  - [AdapterManager - **--impl-- runtime**](#adaptermanager-----impl---runtime)
  - [ManagedSimInputAdapter - **--impl-- runtime**](#managedsiminputadapter-----impl---runtime)
  - [ManagedSimInputAdapter - **--graph-- time**](#managedsiminputadapter-----graph---time)
  - [Example - CSVReader](#example---csvreader)

## Introduction

There are two main categories of writing input adapters, historical and realtime.

When writing historical adapters you will need to implement a "pull" adapter, which pulls data from a historical data source in time order, one event at a time.

There are also ManagedSimAdapters for feeding multiple "managed" pull adapters from a single source (more on that below).

When writing input adapters it is also very important to denote the difference between "graph building time" and "runtime" versions of your adapter.
For example, `csp.adapters.csv` has a `CSVReader` class that is used at graph building time.

**Graph build time components** solely *describe* the adapter.
They are meant to do little else than keep track of the type of adapter and its parameters, which will then be used to construct the actual adapter implementation when the engine is constructed from the graph description.
It is the runtime implementation that actual runs during the engine execution phase to process data.

For clarity of this distinction, in the descriptions below we will denote graph build time components with *--graph--* and runtime implementations with *--impl--*.

## Types of Historical Adapters

There are two flavors of historical input adapters that can be written.
The simplest one is a PullInputAdapter.
A PullInputAdapter can be used to convert a single source into a single timeseries.
The `csp.curve` implementation is a good example of this.
Single source to single timeseries adapters are of limited use however, and the more typical use case is for AdapterManager based input adapters to service multiple InputAdapters from a single source.
For this one would use an AdapterManager to coordinate processing of the data source, and ManagedSimInputAdapter as the individual timeseries providers.

## PullInputAdapter

### PullInputAdapter - Python

To write a Python based `PullInputAdapter` one must write a class that derives from `csp.impl.pulladapter.PullInputAdapter`.
The derived type should the define two methods:

- `def start(self, start_time, end_time)`: this will be called at the start of the engine with the start/end times of the engine.
  `start_Time` and `end_time` will be tz-unaware datetime objects in UTC time.
  At this point the adapter should open its resource and seek to the requested starttime.
- `def next(self)`: this method will be repeatedly called by the engine.
  The adapter should return the next event as a time,value tuple.
  If there are no more events, then the method should return `None`.

The `PullInputAdapter` that you define will be used as the runtime *--impl–-*.
You also need to define a *--graph--* time representation of the time series edge.
In order to do this you should define a `csp.impl.wiring.py_pull_adapter_def`.
The `py_pull_adapter_def` creates a *--graph--* time representation of your adapter:

```python
def py_pull_adapter_def(name, adapterimpl, out_type, **kwargs)
```

- **`name`**: string name for the adapter
- **`adapterimpl`**: a derived implementation of `csp.impl.pulladapter.PullInputAdapter`
- **`out_type`**: the type of the output, should be a `ts[]` type. Note this can use tvar types if a subsequent argument defines the tvar
- **`kwargs`**: \*\*kwargs here be passed through as arguments to the `PullInputAdapter` implementation

Note that the \*\*kwargs passed to `py_pull_adapter_def` should be the names and types of the variables, like `arg1=type1, arg2=type2`.
These are the names of the kwargs that the returned input adapter will take and pass through to the `PullInputAdapter` implementation, and the types expected for the values of those args.

`csp.curve` is a good simple example of this:

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

See example [e2_pullinput.pyy](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e2_pullinput.py) for more details.

### PullInputAdapter - C++

**Step 1)** `PullInputAdapter` impl

Similar to the Python `PullInputAdapter` API is the c++ API which one can leverage to improve performance of an adapter implementation.
The *--impl--* is very similar to python pull adapter.
One should derive from `PullInputAdapter<T>`, a templatized base class (templatized on the type of the timeseries) and define these methods:

- **`start(DateTime start, DateTime end)`**: similar to python API start, called when engine starts.
  Open resource and seek to start time here
- **`stop()`**: called on engine shutdown, cleanup resource
- **`bool next(DateTime & t, T & value)`**: if there is data to provide, sets the next time and value for the adapter and returns true.
  Otherwise, return false

**Step 2)** Expose creator func to python

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
- **`pyengine `**: PyEngine engine wrapper object
- **`pyType`**: this is the type of the timeseries input adapter to be created as a `PyTypeObject`.
  one can switch on this type using switchPyType to create the properly typed instance
- **`pushMode`**: the CSP PushMode for the adapter (pass through to base InputAdapter)
- **`args`**: arguments to pass to the adapter impl

Then simply register the creator method:

**`REGISTER_INPUT_ADAPTER(_my_adapter, create_my_adapter)`**

This will register methodname onto your python module, to be accessed as your module.methodname.
Note this uses `csp/python/InitHelpers` which is used in the `_cspimpl` module.
To do this in a separate python module, you need to register `InitHelpers` in that module.

**Step 3)** Define your *--graph–-* time adapter

One liner now to wrap your impl in a graph time construct using `csp.impl.wiring.input_adapter_def`:

```python
my_adapter = input_adapter_def('my_adapter', my_module._my_adapter, ts[int], arg1=int, arg2={str:'foo'})
```

`my_adapter` can now be called with `arg1, arg2` to create adapters in your graph.
Note that the arguments are typed using `v=t` syntax. `v=(t,default)` is used to define arguments with defaults.

Also note that all input adapters implicitly get a push_mode argument that is defaulted to `csp.PushMode.LAST_VALUE`.

## AdapterManager and ManagedSimInputAdapter - Python

In most cases you will likely want to expose a single source of data into multiple input adapters.
For this use case your adapter should define an AdapterManager *--graph--* time component, and AdapterManagerImpl *--impl--* runtime component.
The AdapterManager *--graph--* time component just represents the parameters needed to create the *--impl--* AdapterManager.
Its the *--impl--* that will have the actual implementation that will open the data source, parse the data and provide it to individual Adapters.

Similarly you will need to define a derived ManagedSimInputAdapter *--impl--* component to handle events directed at an individual time series adapter.

**NOTE** It is highly recommended not to open any resources in the *--graph--* time component.
graph time components can be pruned and/or memoized into a single instance, opening resources at graph time shouldn't be necessary.

### AdapterManager - **--graph-- time**

The graph time AdapterManager doesn't need to derive from any interface.
It should be initialized with any information the impl needs in order to open/process the data source (ie csv file, time column, db connection information, etc etc).
It should also have an API to create individual timeseries adapters.
These adapters will then get passed the adapter manager *--impl--* as an argument where they are created, so that they can register themselves for processing.
The AdapterManager also needs to define a **\_create** method.
The **\_create** is the bridge between the *--graph--* time AdapterManager representation and the runtime *--impl--* object.
**\_create** will be called on the *--graph--* time AdapterManager which will in turn create the *--impl--* instance.
\_create will get two arguments, engine (this represents the runtime engine object that will run the graph) and a memo dict which can optionally be used for any memoization that on might want.

Lets take a look at [`CSVReader`](https://github.com/Point72/csp/blob/main/csp/adapters/csv.py) as an example:

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
  typ denotes the type of the timeseries to create (ie `ts[int]`) and field_map is used for mapping columns onto `csp.Struct` types.
  Note that subscribe returns a `CSVReadAdapter` instance.
  `CSVReadAdapter` is the *--graph--* time representation of the edge (similar to how we defined `csp.curve` above).
  We pass it `self` as its first argument, which will be used to create the AdapterManager *--impl--*
- **`_create`**: the method to create the *--impl--* object from the given *--graph--* time representation of the manager

The `CSVReader` would then be used in graph building code like so:

```python
reader = CSVReader('my_data.csv', time_formatter, symbol_column='SYMBOL', delimiter='|')
# aapl will represent a ts[PriceQuantity] edge that will tick with rows from
# the csv file matching on SYMBOL column AAPL
aapl = reader.subscribe('AAPL', PriceQuantity)
```

### AdapterManager - **--impl-- runtime**

The AdapterManager *--impl--* is responsible for opening the data source, parsing and processing through all the data and managing all the adapters it needs to feed.
The impl class should derive from `csp.impl.adaptermanager.AdapterManagerImpl` and implement the following methods:

- **`start(self,starttime,endtime)`**: this is called when the engine starts up.
  At this point the impl should open the resource providing the data and seek to starttime.
  starttime/endtime will be tz-unaware datetime objects in UTC time
- **`stop(self)`**: this is called at the end of the run, resources should be cleaned up at this point
- **`process_next_sim_timeslice(self, now)`**: this method will be called multiple times through the run.
  The initial call will provide now with starttime.
  The impl's responsibility is to process all data at the given timestamp (more on how to do this below).
  The method should return the next time in the data source, or None if there is no more data to process.
  The method will be called again with the provided timestamp as "now" in the next iteration.
  **NOTE** that process_next_sim_timeslice is required to move ahead in time.
  In most cases the resource data can be supplied in time order, if not it would have to be sorted up front.

`process_next_sim_timeslice` should parse data for a given time/row of data and then push it through to any registered `ManagedSimInputAdapter` that matches on the given row.

### ManagedSimInputAdapter - **--impl-- runtime**

Users will need to define `ManagedSimInputAdapter` derived types to represent the individual timeseries adapter *--impl--* objects.
Objects should derive from `csp.impl.adaptermanager.ManagedSimInputAdapter`.

`ManagedSimInputAdapter.__init__` takes two arguments:

- **`typ`**: this is the type of the timeseries, ie int for a `ts[int]`
- **`field_map`**: Optional, field_map is a dictionary used to map source column names → `csp.Struct` field names.

`ManagedSimInputAdapter` defines a method `push_tick()` which takes the value to feed the input for given timeslice (as defined by "now" at the adapter manager level).
There is also a convenience method called `process_dict()` which will take a dictionary of `{column : value}` entries and convert it properly into the right value based on the given **field_map.**

### ManagedSimInputAdapter - **--graph-- time**

As with the `csp.curve` example, we need to define a graph-time construct that represents a `ManagedSimInputAdapter` edge.
In order to define this we use `py_managed_adapter_def`.
`py_managed_adapter_def` is AdapterManager-"aware" and will properly create the AdapterManager *--impl--* the first time its encountered.
It will then pass the manager impl as an argument to the `ManagedSimInputAdapter`.

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

### Example - CSVReader

Putting this all together lets take a look at a `CSVReader` implementation
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

Here we define CSVReader, our AdapterManager *--graph--* time representation.
It holds the parameters that will be used for the impl, it implements a `subscribe()` call for users to create timeseries and defines a \_create method to create a runtime *--impl–-* instance from the graphtime representation.
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

`CSVReaderImpl` is the runtime *--impl–-*.
It gets created when the engine is being built from the described graph.

- **lines 10-21 - start()**: this is the start method that gets called with the time range the graph will be run against.
  Here we open our resource (`pycsv.DictReader`) and scan t through the data until we reach the requested starttime.

- **lines 23-24 - stop()**: this is the stop call that gets called when the engine is done running and is shutdown, we free our resource here

- **lines 26-29**: the `CSVReader` allows one to subscribe to many symbols from one file.
  symbols are keyed by a provided `SYMBOL` column.
  The individual adapters will self-register with the `CSVReaderImpl` when they are created with the requested symbol.
  `CSVReaderImpl` keeps track of what adapters have been registered for what symbol in its `self._inputs` map.

- **lines 31-43**: this is main method that gets invoked repeatedly throughout the run.
  For every distinct timestamp in the file, this method will get invoked once and the method is expected to go through the resource data for all points with time now, process the row and push the data to any matching adapters.
  The method returns the next timestamp when its done processing all data for "now", or None if there is no more data.
  **NOTE** that the csv impl expects the data to be in time order.
  `process_next_sim_timeslice` must advance time forward.

- **lines 45-49**: this method takes a row of data (provided as a dict from `DictReader`), extracts the symbol and pushes the row through to all input adapters that match

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

- **line 3**: this is where the instance of an adapter *--impl--* registers itself with the `CSVReaderImpl`.
- **line 6+**: this is where we define `CSVReadAdapter`, the *--graph--* time representation of a CSV adapter, returned from `CSVReader.subscribe`

See example [e3_adaptermanager_pullinput.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e3_adaptermanager_pullinput.py) for another example of how to write a managed sim adapter manager.
