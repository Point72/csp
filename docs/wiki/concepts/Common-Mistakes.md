## Table of Contents

- [Graphs vs. nodes](#graphs-vs-nodes)
- [Always use engine time](#always-use-engine-time)

## Graphs vs. nodes

The most common challenge for new `csp` users is understanding the distinction between [graphs](CSP-Graph) and [nodes](CSP-Node). A few issues arise when graphs are treated like nodes and vice versa.

1. *Nodes cannot call other nodes*. Nodes themselves can be considered *atomic* computation units; they perform a single transformation at runtime and do not invoke other nodes to execute.

Example erroneous code

```python
import csp
from csp import ts

@csp.node
def square(x: ts[int]) -> ts[int]:
    return x*x

@csp.node
def cube(x: ts[int]) -> ts[int]:
    return x*square(x) 


csp.run(cube(csp.const(1)), starttime=datetime(2020,1,1), endtime=timedelta())
# the run call above will fail at runtime with the error
# ArgTypeMismatchError: In function square: Expected csp.impl.types.tstype.TsType[int] for argument 'x', got 1 (int)
# square(x) accepts a time-series of int, but when cube is invoked we are passing it the VALUE of x (an int)!
```

Corrected code - we simply make `cube` a graph, not a node.

```python
@csp.graph
def cube(x: ts[int]) -> ts[int]:
    return x*square(x) 
```

2. *Graph code does not access runtime values*. This is the inverse of (1): graph code treats all time-series as Edges and simply "wires" together the application.

Example erroneous code

```python
@csp.graph
def clip(x: ts[int], low: int, high: int) -> ts[int]:
    return min(max(x, low), high)

csp.run(clip(csp.const(3), 1, 4), starttime=datetime(2020,1,1), endtime=timedelta())
# the run fails with - ValueError: boolean evaluation of an edge is not supported
# at graph-time, x is an Edge, not a value: so comparing it to an integer doesn't make sense
# we really want to compare each value itself at runtime
```

Corrected code - we apply the `min` and `max` functions to the values, not the Edge.

```python
@csp.graph
def clip(x: ts[int], low: int, high: int) -> ts[int]:
    return csp.apply(x, lambda u: min(max(u, low), high), int)
```

## Always use engine time

Another common mistake is to use `datetime.now()` inside a `csp.node`. You should **always** use the engine time in a node by calling `csp.now()`, or else your application will not work correctly in historical mode. The *engine time* is the time that the current `csp` engine cycle started. In real-time it will be the wall clock at the time the cycle began, and in historical mode it will be the timestamp of the simulated event. The use of `csp.now()` allows the graph to be run in both execution contexts and still maintain a consistent view of the time.

Example erroneous code

```python
from typing import List

@csp.node
def next_movie_showing(show_times: ts[List[datetime]]) -> ts[datetime]:
    next_showing = None
    for time in show_times:
        if time >= datetime.now(): # list may include some shows today that have already past, so let's filter those out
            if next_showing is None or time < next_showing:
                next_showing = time

    return next_showing
```

If you only run this node in real-time, you may not notice that it has a critical error. When we run on historical data, we will not get any valid show times! The show times in historical data will be on *past* days, but `datetime.now()` will be giving us the *current* time.

Corrected code - we use `csp.now` instead of `datetime.now` so the node works on playback data.

```python
...
    for time in show_times:
            if time >= csp.now():
                ...
```
