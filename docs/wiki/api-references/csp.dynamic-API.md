## csp.dynamic

- **`csp.dynamic(trigger, sub_graph, graph_args...) → csp.DynamicBasket[ ... ]`**
  - **`trigger`**: a `csp.DynamicBasket` input.
    As new keys are added to the basket, they will trigger sub_graph instances to be created.
    As keys are removed, they will shutdown their respective sub-graph
  - **`sub_graph`** - a regular `csp.graph` method that will be wired as new keys are added on trigger
  - **`graph_args`**: these are the args passed to the sub_graph at the time of creation.
    Note the special semantics of argument passing to dynamic sub-graphs:
    - **`scalars`**: can be passed as is, assuming they are known at main graph build time
    - **`timeseries`** - can be passed as is, assuming they are known at main graph build time
    - **`csp.snap(ts)`**: this will convert a timeseries input to a **`scalar`** at the time of graph creation, allowing you to get a "dynamic" scalar value to use at sub_graph build time
    - **`csp.snapkey()`**: this will pass through the key that was added which triggered this dynamic sub-graph.
      One can use this to get the key triggering the sub-graph.
    - **`csp.attach()`**: this will pass through the timeseries of the input trigger for the key which triggered this dynamic sub-graph.
      For example, say we have a dynamic basket of `{ symbol : ts[orders ]}` as our input trigger.
      As a new symbol is added, we will trigger a sub-graph to process this symbol.
      Say we also want to feed in the `ts[orders]` for the given symbol into our sub_graph, we would pass `csp.attach()` as the argument.
  - **`output`**: every output of sub_graph (if there are any) will be returned as a member of a `csp.DynamicBasket` output.
    As new keys are added to the trigger, which generates sub-graphs, keys will be added to the output dynamic basket
    (Note, output keys will only generate on first tick of some output data, not upon instantiation of the sub-graph, since `csp.DynamicBasket` requires all keys to have valid values)

```python
@csp.graph
def my_sub_graph(symbol : str, orders : ts[ Orders ], portfolio_position : ts[int], some_scalar : int) -> ts[Fill]:
   ... regular csp.graph code ...


@csp.graph
def main():
    # position as ts[int]
    portfolio_position = get_portfolio_position()


    all_orders = get_orders()
    # demux fat-pipe of orders into a dynamic basket keyed by symbol
    demuxed_orders = csp.dynamic_demultiplex(all_orders, all_orders.symbol)


    result = csp.dynamic(demuxed_orders, my_sub_graph,
                          csp.snap(all_orders.symbol), # Grab scalar value of all_orders.symbol at time of instantiation
                          #csp.snapkey(),                # Alternative way to grab the key that instantiated the sub-graph
                          csp.attach(),                  # extract the demuxed_orders[symbol] time series of the symbol being created in the sub_graph
                          portfolio_position,            # pass in regular ts[]
                          123)                          # pass in some scalar


    # process result.fills which will be a csp.DynamicBasket of { symbol : ts[Fill] }
```
