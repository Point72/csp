By definition of the graph building code, CSP graphs can only produce acyclical graphs.
However, there are many occasions where a cycle may be required.
For example, lets say you want part of your graph to simulate an exchange.
That part of the graph would need to accept new orders and return acks and executions.
However, the acks / executions would likely need to *feedback* into the same part of the graph that generated the orders.
For this reason, the `csp.feedback` construct exists.
Using `csp.feedback` one can wire a feedback as an input to a node, and effectively bind the actual edge that feeds it later in the graph.
Note that internally the graph is still acyclical.
Internally `csp.feedback` creates a pair of output and input adapters that are bound together.
When a timeseries that is bound to a feedback ticks, it is fed to the feedback which then schedules the tick on its bound input to be executed on the **next engine cycle**.
The next engine cycle will execute with the same engine time as the cycle that generated it, but it will be evaluated in a subsequent cycle.

- **`csp.feedback(ts_type)`**: `ts_type` is the type of the timeseries (ie int, str).
  This returns an instance of a feedback object
  - **`out()`**: this method returns the timeseries edge which can be passed as an input to your node
  - **`bind(ts)`**: this method is called to bind an edge as the source of the feedback after the fact

A simple example should help demonstrate a possible usage.
Lets say we want to simulate acking orders that are generated from a node called `my_algo`.
In addition to generating the orders, `my_algo` also wants needs to receive the execution reports (this is demonstrated in example `e_13_feedback.py`)

The graph code would look something like this:

```python
# Simulate acking an order
@csp.node
def my_exchange(order:ts[Order]) -> ts[ExecReport]:
    # ... impl details ...

@csp.node
def my_algo(exec_report:ts[ExecReport]) -> ts[Order]:
    # .. impl details ...

@csp.graph
def my_graph():
    # create the feedback first so that we can refer to it later
    exec_report_fb = csp.feedback(ExecReport)

    # generate orders, passing feedback out() which isn't bound yet
    orders = my_algo(exec_report_fb.out())

    # get exec_reports from "simulator"
    exec_report = my_exchange(orders)

    # now bind the exec reports to the feedback, finishing the "loop"
    exec_report_fb.bind(exec_report)
```

The graph would end up looking like this.
It remains acyclical, but the `FeedbackOutputDef` is bound to the `FeedbackInputDef` here, any tick to out will push the tick to in on the next cycle:

![366521848](https://github.com/Point72/csp/assets/3105306/c4f920ff-49f9-4a52-8404-7c1989768da7)
