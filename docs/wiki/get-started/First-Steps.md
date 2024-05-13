When writing CSP code there will be runtime components in the form of `csp.node` methods, as well as graph-building components in the form of `csp.graph` components.

It is important to understand that `csp.graph` components will only be executed once at application startup in order to construct the graph.
Once the graph is constructed, `csp.graph` code is no longer needed.
Once the graph is run, only inputs, `csp.node`s and outputs will be active as data flows through the graph, driven by input ticks.

For example, this is a simple bit of graph code:

```python
import csp
from csp import ts
from datetime import datetime


@csp.node
def spread(bid: ts[float], ask: ts[float]) -> ts[float]:
    if csp.valid(bid, ask):
        return ask - bid


@csp.graph
def my_graph():
    bid = csp.const(1.0)
    ask = csp.const(2.0)
    bid = csp.multiply(bid, csp.const(4))
    ask = csp.multiply(ask, csp.const(3))
    s = spread(bid, ask)

    csp.print('spread', s)
    csp.print('bid', bid)
    csp.print('ask', ask)


if __name__ == '__main__':
    csp.run(my_graph, starttime=datetime.utcnow())
```

In order to help visualize this graph, you can call `csp.show_graph`:

![359407708](https://github.com/Point72/csp/assets/3105306/8cc50ad4-68f9-4199-9695-11c136e3946c)

The result of this would be:

```
2020-04-02 15:33:38.256724 bid:4.0
2020-04-02 15:33:38.256724 ask:6.0
2020-04-02 15:33:38.256724 spread:2.0
```
