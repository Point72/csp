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

TODO

See example [e3_adaptermanager_pullinput.py](https://github.com/Point72/csp/blob/main/examples/04_writing_adapters/e3_adaptermanager_pullinput.py) for another example of how to write a managed sim adapter manager.
