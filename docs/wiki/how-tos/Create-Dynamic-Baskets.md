CSP graphs are somewhat limiting in that they cannot change shape once the process starts up.
CSP dynamic graphs addresses this issue by introducing a construct to allow applications to dynamically add / remove sub-graphs from a running graph.

`csp.DynamicBasket`s are a pre-requisite construct needed for dynamic graphs.
`csp.DynamicBasket`s work just like regular static CSP baskets, however dynamic baskets can change their shape over time.
`csp.DynamicBasket`s can only be created from either CSP nodes or from `csp.dynamic` calls, as described below.
A node can take a `csp.DynamicBasket` as an input or generate a dynamic basket as an output.
Dynamic baskets are always dictionary-style baskets, where time series can be added by key.
Note that timeseries can also be removed from dynamic baskets.

## Syntax

Dynamic baskets are denoted by the type `csp.DynamicBasket[key_type, ts_type]`, so for example `csp.DynamicBasket[str,int]` would be a dynamic basket that will have keys of type str, and timeseries of type int.
One can also use the non-python shorthand `{ ts[str] : ts[int] }` to signify the same.

## Generating dynamic basket output

For nodes that generate dynamic basket output, they would use the same interface as regular basket outputs.
The difference being that if you output a key that hasn't been seen before, it will automatically be added to the dynamic basket.
In order to remove a key from a dynamic basket output, you would use the `csp.remove_dynamic_key` method.
**NOTE** that it is illegal to add and remove a key in the same cycle:

```python
@csp.node
def dynamic_demultiplex_example(data : ts[ 'T' ], key : ts['K']) -> csp.DynamicBasket['T', 'K']:
    if csp.ticked(data) and csp.valid(key):
        csp.output({ key : data })


        ## To remove a key, which wouldn't be done in this example node:
        ## csp.remove_dynamic_key(key)
```

To remove a key one would use `csp.remove_dynamic_key`.
For a single unnamed output, the method expects the key.
For named outputs, the arguments would be `csp.remove_dynamic_key(output_name, key)`

## Consuming dynamic basket input

Taking dynamic baskets as input is exactly the same as static baskets.
There is one additional bit of information available on dynamic basket inputs though, which is the .shape property.
As keys are added or removed, the `basket.shape` property will tick the the change events.
The `.shape` property behaves effectively as a `ts[csp.DynamicBasketEvents]`:

```python
@csp.node
def consume_dynamic_basket(data : csp.DynamicBasket[str,int]):
     if csp.ticked(data.shape):
         for key in data.shape.added:
             print(f'key {key} was added')
         for key in data.shape.removed:
             print(f'key {key} was removed')


      if csp.ticked(data):
          for key,value in data.tickeditems():
             #...regular basket access here
```
