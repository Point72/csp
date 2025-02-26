## Table of Contents

- [Table of Contents](#table-of-contents)
- [Anatomy of a `csp.node`](#anatomy-of-a-cspnode)
- [Basket inputs](#basket-inputs)
- [Node Outputs](#node-outputs)
- [Basket Outputs](#basket-outputs)
- [Generic Types](#generic-types)

## Anatomy of a `csp.node`

The heart of a calculation graph are the `csp.nodes` that run the computations.
`csp.node` methods can take any number of scalar and timeseries arguments, and can return 0 → N timeseries outputs.
Timeseries inputs/outputs should be thought of as the edges that connect components of the graph.
These "edges" can tick whenever they have a new value.
Every tick is associated with a value and the time of the tick.
`csp.nodes` can have various other features, here is a an example of a `csp.node` that demonstrates many of the features.
Keep in mind that nodes will execute repeatedly as inputs tick with new data.
They may (or may not) generate an output as a result of an input tick.

```python
from datetime import timedelta

@csp.node(name='my_node')                                                # 1
def demo_node(n: int, xs: ts[float], ys: ts[float]) -> ts[float]:        # 2
    with csp.alarms():                                                   # 3
        # Define an alarm time-series of type bool                       # 4
        alarm = csp.alarm(bool)                                          # 5
                                                                         # 6
    with csp.state():                                                    # 7
        # Create a state variable bound to the node                      # 8
        s_sum = 0.0                                                      # 9
                                                                         # 10
    with csp.start():                                                    # 11
        # Code block that executes once on start of the engine           # 12
        # one can set timeseries properties here as well, such as        # 13
        # csp.set_buffering_policy(xs, tick_count=5)                     # 14
        # csp.set_buffering_policy(xs, tick_history=timedelta(minutes=1))# 15
        # csp.make_passive(xs)                                           # 16
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)            # 17
                                                                         # 18
    with csp.stop():                                                     # 19
        pass  # code block to execute when the engine is done            # 20
                                                                         # 21
    if csp.ticked(xs, ys) and csp.valid(xs, ys):                         # 22
        s_sum += xs * ys                                                 # 23
                                                                         # 24
    if csp.ticked(alarm):                                                # 25
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)            # 26
        return s_sum                                                     # 27
```

Lets review line by line

1\) Every CSP node must start with the **`@csp.node`** decorator. The name of the node will be the name of the function, unless a `name` argument is provided. The name is used when visualizing a graph with `csp.show_graph` or profiling with CSP's builtin [`profiler`](#Profile-csp-code).

2\) CSP nodes are fully typed and type-checking is strictly enforced.
All arguments must be typed, as well as all outputs.
Outputs are typed using function annotation syntax.

Single outputs can be unnamed, for multiple outputs they must be named.
When using multiple outputs, annotate the type using **`def my_node(inputs) → csp.Outputs(name1=ts[<T>], name2=ts[<V>])`** where `T` and `V` are the respective types of `name1` and `name2`.

Note the syntax of timeseries inputs, they are denoted by **`ts[type]`**.
Scalars can be passed in as regular types, in this example we pass in `n` which expects a type of `int`

3\) **`with csp.alarms()`**: nodes can (optionally) declare internal alarms, every instance of the node will get its own alarm that can be scheduled and act just like a timeseries input.
All alarms must be declared within the alarms context.

5\) Instantiate an alarm in the alarms context using the `csp.alarm(typ)` function. This creates an alarm which is a time-series of type `typ`.

7\) **`with csp.state()`**: optional state variables can be defined under the state context.
Note that variables declared in state will live across invocations of the method.

9\) An example declaration and initialization of state variable `s_sum`.
It is good practice to name state variables prefixed with `s_`, which is the convention in the CSP codebase.

11\) **`with csp.start()`**: an optional block to execute code at the start of the engine.
Generally this is used to setup initial timers or set input timeseries properties such as buffer sizes, or to make inputs passive

14-15) **`csp.set_buffering_policy`**: nodes can request a certain amount of history be kept on the incoming time series, this can be denoted in number of ticks or in time.
By setting a buffering policy, nodes can access historical values of the timeseries (by default only the last value is kept)

16\) **`csp.make_passive`** / **`csp.make_active`**: Nodes may not need to react to all of their inputs, they may just need their latest value.
For performance purposes the node can mark an input as passive to avoid triggering the node unnecessarily.
`make_active` can be called to reactivate an input.

17\) **`csp.schedule_alarm`**: scheduled a one-shot tick on the given alarm input.
The values given are the timedelta before the alarm triggers and the value it will have when it triggers.
Note that `schedule_alarm` can be called multiple times on the same alarm to schedule multiple triggers.

19\) **`with csp.stop()`** is an optional block that can be called when the engine is done running.

22\) all nodes will have if conditions to react to different inputs.
**`csp.ticked()`** takes any number of inputs and returns true if **any** of the inputs ticked.
**`csp.valid()`** similar takes any number of inputs however it only returns true if **all** inputs are valid.
Valid means that an input has had at least one tick and so it has a "current value".

23\) One of the benefits of CSP is that you always have easy access to the latest value of all inputs.
`xs` and `ys` on line 22,23 will always have the latest value of both inputs, even if only one of them just ticked.

25\) This demonstrates how an alarm can be treated like any other input.

27\) We tick our running "sum" as an output here every second.

## Basket inputs

In addition to single time-series inputs, a node can also accept a **basket** of time series as an argument.
A basket is essentially a collection of timeseries which can be passed in as a single argument.
Baskets can either be list baskets or dict baskets.
Individual timeseries in a basket can tick independently, and they can be looked at and reacted to individually or as a collection.

For example:

```python
@csp.node                                      # 1
def demo_basket_node(                          # 2
    list_basket: [ts[int]],                    # 3
    dict_basket: {str: ts[int]}                # 4
) -> ts[float]:                                # 5
                                               # 6
    if csp.ticked(list_basket):                # 7
        return sum(list_basket.validvalues())  # 8
                                               # 9
    if csp.ticked(list_basket[3]):             # 10
        return list_basket[3]                  # 11
                                               # 12
    if csp.ticked(dict_basket):                # 13
        # can iterate over ticked key,items    # 14
        # for k,v in dict_basket.tickeditems():# 15
        #     ...                              # 16
        return sum(dict_basket.tickedvalues()) # 17
```

3\) Note the syntax of basket inputs.
list baskets are noted as `[ts[type]]` (a list of time series) and dict baskets are `{key_type: ts[ts_type]}` (a dictionary of timeseries keyed by type `key_type`). It is also possible to use the `List[ts[int]]` and `Dict[str, ts[int]]` typing notation.

7\) Just like single timeseries, we can react to a basket if it ticked.
The convention is the same as passing multiple inputs to `csp.ticked`, `csp.ticked` is true if **any** basket input ticked.
`csp.valid` is true is **all** basket inputs are valid.

8\) baskets have various iterators to access their inputs:

- **`tickedvalues`**: iterator of values of all ticked inputs
- **`tickedkeys`**: iterator of keys of all ticked inputs (keys are list index for list baskets)
- **`tickeditems`**: iterator of (key,value) tuples of ticked inputs
- **`validvalues`**: iterator of values of all valid inputs
- **`validkeys`**: iterator of keys of all valid inputs
- **`validitems`**: iterator of (key,value) tuples of valid inputs
- **`keys`**: list of keys on the basket (**dictionary baskets only** )

10-11) This demonstrates the ability to access an individual element of a
basket and react to it as well as access its current value

## **Node Outputs**

Nodes can return any number of outputs (including no outputs, in which case it is considered an "output" or sink node,
see [Graph Pruning](CSP-Graph#graph-pruning)).
Nodes with single outputs can return the output as an unnamed output.
Nodes returning multiple outputs must have them be named.
When a node is called at graph building time, if it is a single unnamed node the return variable is an edge representing the output which can be passed into other nodes.
An output timeseries cannot be ticked more than once in a given node invocation.
If the outputs are named, the return value is an object with the outputs available as attributes.
For example (examples below demonstrate various ways to output the data as well)

```python
@csp.node
def single_unnamed_outputs(n: ts[int]) -> ts[int]:
    # can either do
    return n
    # or
    # csp.output(n) to continue processes after output


@csp.node
def multiple_named_outputs(n: ts[int]) -> csp.Outputs(y=ts[int], z=ts[float]):
    # can do
    # csp.output(y=n, z=n+1.) to output to multiple outputs
    # or separate the outputs to tick out at separate points:
    # csp.output(y=n)
    # ...
    # csp.output(z=n+1.)
    # or can return multiple values with:
    return csp.output(y=n, z=n+1.)

@csp.graph
def my_graph(n: ts[int]):
    x = single_unnamed_outputs(n)
    # x represents the output edge of single_unnamed_outputs,
    # we can pass it a time series input to other nodes
    csp.print('x', x)


    result = multiple_named_outputs(n)
    # result holds all the outputs of multiple_named_outputs, which can be accessed as attributes
    csp.print('y', result.y)
    csp.print('z', result.z)
```

## Basket Outputs

Similarly to inputs, a node can also produce a basket of timeseries as an output.
For example:

```python
class MyStruct(csp.Struct):                                               # 1
    symbol: str                                                           # 2
    index: int                                                            # 3
    value: float                                                          # 4
                                                                          # 5
@csp.node                                                                 # 6
def demo_basket_output_node(                                              # 7
    in_: ts[MyStruct],                                                    # 8
    symbols: [str],                                                       # 9
    num_symbols: int                                                      # 10
) -> csp.Outputs(                                                         # 11
    dict_basket=csp.OutputBasket({str: ts[float]}, shape="symbols"),  # 15
    list_basket=csp.OutputBasket([ts[float]], shape="num_symbols"),   # 16
):                                                                        # 17
                                                                          # 18
    if csp.ticked(in_):                                                   # 19
        # output to dict basket                                           # 20
        csp.output(dict_basket[in_.symbol], in_.value)                    # 21
        # alternate output syntax, can output multiple keys at once       # 22
        # csp.output(dict_basket={in_.symbol: in_.value})                 # 23
        # output to list basket                                           # 24
        csp.output(list_basket[in_.index], in_.value)                     # 25
        # alternate output syntax, can output multiple keys at once       # 26
        # csp.output(list_basket={in_.index: in_.value})                  # 27
```

11-17) Note the output declaration syntax.
A basket output can be either named or unnamed (both examples here are named), and its shape can be specified two ways.
The `shape` parameter is used with a scalar value that defines the shape of the basket, or the name of the scalar argument (a dict basket expects shape to be a list of keys. lists basket expects `shape` to be an `int`).
`shape_of` is used to take the shape of an input basket and apply it to the output basket.

20+) There are several choices for output syntax.
The following work for both list and dict baskets:

- `csp.output(basket={key: value, key2: value2, ...})`
- `csp.output(basket[key], value)`
- `csp.output({key: value}) # only works if the basket is the only output`

## Generic Types

CSP supports syntax for generic types as well.
To denote a generic type we use a string (typically `'T'` is used) to denote a generic type.
When a node is called the type of the argument will get bound to the given type variable, and further inputs / outputs will be checked and bound to said typevar.
Note that the string syntax `'~T'` denotes the argument expects the *value* of a type, rather than a type itself:

```python
@csp.node
def sample(trigger: ts[object], x: ts['T']) -> ts['T']:
    '''will return current value of x on trigger ticks'''
    with csp.state():
        csp.make_passive(x)

    if csp.ticked(trigger) and csp.valid(x):
        return x


@csp.node
def const(value: '~T') -> ts['T']:
    ...
```

`sample` takes a timeseries of type `'T'` as an input, and returns a timeseries of type `'T'`.
This allows us to pass in a `ts[int]` for example, and get a `ts[int]` as an output, or `ts[bool]` → `ts[bool]`

`const` takes value as an *instance* of type `T`, and returns a timeseries of type `T`.
So we can call `const(5)` and get a `ts[int]` output, or `const('hello!')` and get a `ts[str]` output, etc...

If a value is provided rather than an explicit type argument (for example, to `const`) then CSP resolves the type using internal logic. In some cases, it may be easier to override the automatic type inference.
Users can force a type variable to be a specific value with the `.using` function. For example, `csp.const(1)` will be resolved to a `ts[int]`; if you want to instead force the type to be `float`, do `csp.const.using(T=float)(1)`.
