## Table of Contents

- [Table of Contents](#table-of-contents)
- [Anatomy of a `csp.graph`](#anatomy-of-a-cspgraph)
- [Graph Propagation and Single-dispatch](#graph-propagation-and-single-dispatch)
- [Graph Pruning](#graph-pruning)
- [Collecting Graph Outputs](#collecting-graph-outputs)

## Anatomy of a `csp.graph`

To reiterate, csp.graph methods are called in order to construct the graph and are only executed before the engine is run.
csp.graph methods don't do anything special, they are essentially regular python methods, but they can be defined to accept inputs and generate outputs similar to csp.nodes.
This is solely used for type checking.
csp.graph methods can be created to encapsulate components of a graph, and can be called from other csp.graph methods in order to help facilitate graph building.

Simple example:

```python
@csp.graph
def calc_symbol_pnl(symbol: str, trades: ts[Trade]) -> ts[float]:
    # sub-graph code needed to compute pnl for given symbol and symbol's trades
    # sub-graph can subscribe to market data for the symbol as needed
    ...


@csp.graph
def calc_portfolio_pnl(symbols: [str]) -> ts[float]:
    symbol_pnl = []
    for symbol in symbols:
        symbol_trades = trade_adapter.subscribe(symbol)
        symbol_pnl.append(calc_symbol_pnl(symbol, symbol_trades))

    return csp.sum(symbol_pnl)
```

In this simple example we have a csp.graph component `calc_symbol_pnl` which encapsulates computing pnl for a single symbol.
`calc_portfolio_pnl` is a graph that computes portfolio level pnl, it invokes the symbol-level pnl calc for every symbol, then sums up the results for the portfolio level pnl.

## Graph Propagation and Single-dispatch

The `csp` graph propagation algorithm ensures that all nodes are executed *once* per engine cycle, and in the correct order.
Correct order means, that all input dependencies of a given node are guaranteed to have been evaluated before a given node is executed.
Take this graph for example:

![359407953](https://github.com/Point72/csp/assets/3105306/d9416353-6755-4e37-8467-01da516499cf)

On a given cycle lets say the `bid` input ticks.
The `csp` engine will ensure that **`mid`** is executed, followed by **`spread`** and only once **`spread`**'s output is updated will **`quote`** be called.
When **`quote`** executes it will have the latest values of the `mid` and `spread` calc for this cycle.

## Graph Pruning

One should note a subtle optimization technique in `csp` graphs.
Any part of a graph that is created at graph building time, but is NOT connected to any output nodes, will be pruned from the graph and will not exist during runtime.
An output is defined as either an output adapter or a `csp.node` without any outputs of its own.
The idea here is that we can avoid doing work if it doesn't result in any output being generated.
In general its best practice for all csp.nodes to be \***side-effect free**, in other words they shouldn't mutate any state outside of the node.
Assuming all nodes are side-effect free, pruning the graph would not have any noticeable effects.

## Collecting Graph Outputs

If the `csp.graph` passed to `csp.run` has outputs, the full timeseries will be returned from `csp.run` like so:

**outputs example**

```python
import csp
from datetime import datetime, timedelta

@csp.graph
def my_graph() -> ts[int]:
    return csp.merge(csp.const(1), csp.const(2, timedelta(seconds=1)))

if __name__ == '__main__':
    res = csp.run(my_graph, starttime=datetime(2021,11,8))
    print(res)
```

result:

```raw
{0: [(datetime.datetime(2021, 11, 8, 0, 0), 1), (datetime.datetime(2021, 11, 8, 0, 0, 1), 2)]}
```

Note that the result is a list of `(datetime, value)` tuples.

You can also use [csp.add_graph_output](<https://github.com/Point72/csp/wiki/1.-Generic-Nodes-(csp.baselib)#adapters>) to add outputs.
These do not need to be in the top-level graph called directly from `csp.run`.

This gives the same result:

**add_graph_output example**

```python
@csp.graph
def my_graph():
    csp.add_graph_output('a', csp.merge(csp.const(1), csp.const(2, timedelta(seconds=1))))
```

In addition to python outputs like above, you can set the optional `csp.run` argument `output_numpy` to `True` to get outputs as numpy arrays:

**numpy outputs**

```python
result = csp.run(my_graph, starttime=datetime(2021,11,8), output_numpy=True)
```

result:

```raw
{0: (array(['2021-11-08T00:00:00.000000000', '2021-11-08T00:00:01.000000000'], dtype='datetime64[ns]'), array([1, 2], dtype=int64))}
```

Note that the result there is a tuple per output, containing two numpy arrays, one with the datetimes and one with the values.
