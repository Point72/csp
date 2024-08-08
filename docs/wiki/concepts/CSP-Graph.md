## Table of Contents

- [Table of Contents](#table-of-contents)
- [Anatomy of a `csp.graph`](#anatomy-of-a-cspgraph)
- [Graph Propagation and Single-dispatch](#graph-propagation-and-single-dispatch)
- [Graph Pruning](#graph-pruning)
- [Collecting Graph Outputs](#collecting-graph-outputs)

## Anatomy of a `csp.graph`

`csp.graph` methods are called in order to construct the graph and are only executed before the engine is run. A graph is a collection of nodes and adapters which can either be executed as an argument to `csp.run` or composed into a larger graph.
The `csp.graph` decorator is only used for type validation and it is optional when creating a CSP program. A standard Python function without the decorator can also be passed as an argument to `csp.run` if type validation is not required.
`csp.graph` methods can be created to encapsulate components of a graph, and can be called from other `csp.graph` methods in order to help facilitate graph building.

Simple example:

```python
@csp.graph
def calc_user_time(session_data: ts[UserSession]) -> ts[float]:
    # sub-graph code needed to compute the time a user spends on a website
    session_time = session_data.logout_time - session_data.login_time
    time_online = csp.stats.sum(session_time)
    return time_online


@csp.graph
def calc_site_traffic(users: List[str]) -> ts[float]:
    user_time = []
    for user in users:
        user_sessions = get_session(user)
        user_time.append(calc_user_time(user_sessions))

    return csp.sum(user_time)
```

In this simple example we compute the total time all users spend on a website. We have a `csp.graph` subcomponent `calc_user_time` which computes the time a single user spends on the site throughout the run.
Then, in `calc_site_traffic` we compute the total user traffic by creating the user-level subgraph for each account and aggregating the results.

## Graph Propagation and Single-Dispatch

The CSP graph propagation algorithm ensures that all nodes are executed *after* any of their dependencies on a given engine cycle.

> \[!IMPORTANT\]
> An *engine cycle* refers to a single execution of a CSP graph. There can be multiple engine cycles at the same *timestamp*; for example, a single data source may have two events both at `2020-01-01 00:00:00`. These events will be executed in two *cycles* that both occur at the same timestamp. Another case where multiple cycles can occur is [csp.feedback](Add-Cycles-in-Graphs).

For example, consider the graph below:

![359407953](https://github.com/Point72/csp/assets/3105306/d9416353-6755-4e37-8467-01da516499cf)

Individuals nodes are executed in *rank order* where the rank of a node is defined as the longest path between the node and an input adapter. The "mid" node is at rank 1, while "spread" is at rank 2 and "quote" is rank 3. Therefore, if "bid" ticks on a given engine cycle then "mid" will be executed before "spread" and "quote". Note that the order of node execution *within* a rank is undefined, and users should never rely on the execution order of nodes at the same rank.

## Graph Pruning

Any node in a graph that is not connected to an output will be pruned from the graph and will not exist during runtime.
An output is defined as either an output adapter or a `csp.node` without any outputs of its own.
Pruning is an optimization which avoids executing nodes whose result will be discarded.
As a result, it's best practice for any `csp.node` to be \***side-effect free**; they shouldn't mutate any state outside of the node.

## Executing a Graph

Graphs can be executed using the `csp.run` function. Execution takes place in either real-time or historical mode (see [Execution Modes](Execution-Modes)) depending on the `realtime` argument. Graph execution begin at a `starttime` and ends at an `endtime`; the `endtime` argument can either be a `datetime` which is past the start *or* a `timedelta` which is the duration of the run. For example, if we wish to run our `calc_site_traffic` graph over one week of historical data we can execute it with:

```python
csp.run(calc_site_traffic, users=['alice', 'bob'], starttime=start, endtime=timedelta(weeks=1), realtime=False)
```

## Collecting Graph Outputs

There are multiple methods of getting in-process outputs after executing a `csp.graph`. If the graph returns one or more time-series, the full history of those values will be returned from `csp.run`.

**return example**

```python
import csp
from datetime import datetime, timedelta

@csp.graph
def my_graph() -> ts[int]:
    return csp.merge(csp.const(1), csp.const(2, delay=timedelta(seconds=1)))

res = csp.run(my_graph, starttime=datetime(2021,11,8))
```

res:

```raw
{0: [(datetime.datetime(2021, 11, 8, 0, 0), 1), (datetime.datetime(2021, 11, 8, 0, 0, 1), 2)]}
```

Note that the result is a list of `(time, value)` tuples. You can have the result returned as two separate NumPy arrays, one for the times and one for the values, by setting `output_numpy=True` in the `run` call.

```python
res = csp.run(my_graph, starttime=datetime(2021,11,8), output_numpy=True)
```

res:

```raw
{0: (array(['2021-11-08T00:00:00.000000000', '2021-11-08T00:00:01.000000000'], dtype='datetime64[ns]'), array([1, 2], dtype=int64))}
```

You can also use [csp.add_graph_output](Base-Adapters-API#cspadd_graph_output) to add outputs.
These do not need to be in the top-level graph called directly from `csp.run`. Users can also specify the amount of history they want stored in the output using the `tick_count` and `tick_history` arguments to `add_graph_output`. For example, if only the last value needs to be stored set `tick_count=1`.

**add_graph_output example**

```python
@csp.graph
def my_graph():
    same_thing = csp.merge(csp.const(1), csp.const(2, delay=timedelta(seconds=1)))
    csp.add_graph_output('my_name', same_thing)

res = csp.run(my_graph, starttime=datetime(2021,11,8))
```

res:

```raw
{'my_name': [(datetime.datetime(2021, 11, 8, 0, 0), 1), (datetime.datetime(2021, 11, 8, 0, 0, 1), 2)]}
```
