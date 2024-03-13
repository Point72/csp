Edges in csp contain some methods to serve as syntactic sugar for stringing nodes together in a pipeline. This makes it easier to read/modify workflows and avoids the need for nested brackets.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [`apply`](#apply)
- [`pipe`](#pipe)
- [`run`](#run)
- [Example of functional methods](#example-of-functional-methods)

## `apply`

```python
Edge.apply(self, func, *args, **kwargs)
```
Calls `csp.apply` on the edge with the provided python `func`.

Args:
- **`func`**: A scalar function that will be applied on each value of the Edge. If a different output type is returned, pass a tuple `(f, typ)`, where `typ` is the output type of f
- **`args`**: Positional arguments passed into `func`
- **`kwargs`**: Dictionary of keyword arguments passed into func

## `pipe`
```python
Edge.pipe(self, node, *args, **kwargs)
```
Calls the `node` on the edge.

Args:
- **`node`**: A graph node that will be applied to the Edge, which is passed into node as the first argument.
  Alternatively, a `(node, edge_keyword)` tuple where `edge_keyword` is a string indicating the keyword of node that expects the edge.
- **`args`**: Positional arguments passed into `node`
- **`kwargs`**: Dictionary of keyword arguments passed into `node`

## `run`
```python
Edge.run(self, node, *args, **kwargs)
```

Alias for `csp.run(self, *args, **kwargs)`

## Example of functional methods

```python
import csp
from datetime import datetime, timedelta
import math

(csp.timer(timedelta(minutes=1))
    .pipe(csp.count)
    .pipe(csp.delay, timedelta(seconds=1))
    .pipe((csp.sample, 'x'), trigger=csp.timer(timedelta(minutes=2)))
    .apply((math.sin, float))
    .apply(math.pow, 3)
    .pipe(csp.firstN, 10)
    .run(starttime=datetime(2000,1,1), endtime=datetime(2000,1,2)))

```
