`csp.baselib` defines some generally useful adapters, which are also imported directly into the CSP namespace when importing CSP.

These are all graph-time constructs.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [`csp.timer`](#csptimer)
- [`csp.const`](#cspconst)
- [`csp.curve`](#cspcurve)
- [`csp.add_graph_output`](#cspadd_graph_output)
- [`csp.feedback`](#cspfeedback)

## `csp.timer`

```python
csp.timer(
    interval: timedelta,
    value: '~T' = True,
    allow_deviation: bool = False
)
```

This will create a repeating timer edge that will tick on the given `timedelta` with the given value (value defaults to `True`, returning a `ts[bool]`)

Args:

- **`interval`**: how often to tick value
- **`value`**: the actual value that will tick every interval (defaults to the value `True`)
- **`allow_deviation`**: When running in realtime the engine will ensure timers execute exactly when they requested on their intervals.
  If your engine begins to lag, timers will still execute at the expected time "in the past" as the engine catches up
  (imagine having a `csp.timer` fire every 1/2 second but the engine becomes delayed for 1 second.
  By default the half seconds will still execute until time catches up to wallclock).
  When `allow_deviation` is `True`, and the engine is in realtime mode, subsequent timers will always be scheduled from the current wallclock + interval,
  so they won't end up lagging behind at the expensive of the timer skewing.

## `csp.const`

```python
csp.const(
    value: '~T',
    delay: timedelta = timedelta()
)
```

This will create an edge that ticks one time with the value provided.
By default this will tick at the start of the engine, delta can be provided to delay the tick

## `csp.curve`

```python
csp.curve(
    typ: 'T',
    data: typing.Union[list, tuple]
)
```

This allows you to convert a list of non-CSP data into a ticking edge in CSP

Args:

- **`typ`**: is the type of the value of the data of this edge
- **`data`**: is either a list of tuples of `(datetime, value)`, or a tuple of two equal-length numpy ndarrays, the first with datetimes and the second with values.
  In either case, that will tick on the returned edge into the engine, and the data must be in time order.
  Note that for the list of tuples case, you can also provide tuples of (timedelta, value) where timedelta will be the offset from the engine's start time.

## `csp.add_graph_output`

```python
csp.add_graph_output(
    key: object,
    input: ts['T'],
    tick_count: int = -1,
    tick_history: timedelta = timedelta()
)
```

This allows you to connect an edge as a "graph output".
All edges added as outputs will be returned to the caller from `csp.run` as a dictionary of `key: [(datetime, value)]`
(list of datetime, values that ticked on the edge) or if `csp.run` is passed `output_numpy=True`, as a dictionary of
`key: (array, array)` (tuple of two numpy arrays, one with datetimes and one with values).
See [Collecting Graph Outputs](CSP-Graph#collecting-graph-outputs)

Args:

- **`key`**: key to return the results as from `csp.run`
- **`input`**: edge to connect
- **`tick_count`**: number of ticks to keep in the buffer (defaults to -1 - all ticks)
- **`tick_history`**: amount of ticks to keep by time window (defaults to keeping all history)

## `csp.feedback`

```python
csp.feedback(typ)
```

`csp.feedback` is a construct that can be used to create artificial loops in the graph.
Use feedbacks in order to delay bind an input to a node in order to be able to create a loop
(think of writing a simulated exchange that takes orders in and needs to feed responses back to the originating node).

`csp.feedback` itself is not an edge, its a construct that allows you to access the delayed edge / bind a delayed input.

Args:

- **`typ`**: type of the edge's data to be bound

Methods:

- **`out()`**: call this method on the feedback object to get the edge which can be wired as an input
- **`bind(x: ts[object])`**: call this to bind an edge to the feedback
