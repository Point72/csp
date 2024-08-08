CSP comes with some basic constructs readily available and commonly used.
The latest set of base nodes can be found in the `csp.baselib` module.

All of the nodes noted here are imported directly into the CSP namespace when importing CSP.

These are all graph-time constructs.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [`csp.print`](#cspprint)
- [`csp.log`](#csplog)
- [`csp.sample`](#cspsample)
- [`csp.firstN`](#cspfirstn)
- [`csp.count`](#cspcount)
- [`csp.delay`](#cspdelay)
- [`csp.diff`](#cspdiff)
- [`csp.merge`](#cspmerge)
- [`csp.split`](#cspsplit)
- [`csp.filter`](#cspfilter)
- [`csp.drop_dups`](#cspdrop_dups)
- [`csp.unroll`](#cspunroll)
- [`csp.collect`](#cspcollect)
- [`csp.flatten`](#cspflatten)
- [`csp.default`](#cspdefault)
- [`csp.gate`](#cspgate)
- [`csp.apply`](#cspapply)
- [`csp.null_ts`](#cspnull_ts)
- [`csp.stop_engine`](#cspstop_engine)
- [`csp.multiplex`](#cspmultiplex)
- [`csp.demultiplex`](#cspdemultiplex)
- [`csp.dynamic_demultiplex`](#cspdynamic_demultiplex)
- [`csp.dynamic_collect`](#cspdynamic_collect)
- [`csp.drop_nans`](#cspdrop_nans)
- [`csp.times`](#csptimes)
- [`csp.times_ns`](#csptimes_ns)
- [`csp.accum`](#cspaccum)
- [`csp.exprtk`](#cspexprtk)

## `csp.print`

```python
csp.print(
    tag: str,
    x: ts['T'])
```

This node will print (using python `print()`) the time, tag and value of `x` for every tick of `x`

## `csp.log`

```python
csp.log(
    level: int,
    tag: str,
    x: ts['T'],
    logger: typing.Optional[logging.Logger] = None,
    logger_tz: object = None,
    use_thread: bool = False
)
```

Similar to `csp.print`, this will log ticks using the logger on the provided level.
The default CSP logger is used if none is provided to the node.

Args:

- **`logger_tz`**: time zone to use for log entries
- **`use_thread`**: if `True`, the logging calls will occur in a separate thread as to not block graph execution.
  This can be useful when printing large strings in log calls.
  If individual time-series values are subject to modification *after* the log call, then the user must pass in a copy of the time-series if they wish to have proper threaded logging.

## `csp.sample`

```python
csp.sample(
    trigger: ts[object],
    x: ts['T']
) → ts['T']
```

Use this to down-sample an input.
`csp.sample` will return the current value of `x` any time trigger ticks.
This can be combined with `csp.timer` to sample the input on a time interval.

## `csp.firstN`

```python
csp.firstN(
    x: ts['T'],
    N: int
) → ts['T']
```

Only output the first `N` ticks of the input.

## `csp.count`

```python
csp.count(x: ts[object]) → ts[int]
```

Returns the ticking count of ticks of the input

## `csp.delay`

```python
csp.delay(
    x: ts['T'],
    delay: typing.Union[timedelta, int]
) → ts['T']
```

This will delay all ticks of the input `x` by the given `delay`, which can be given as a `timedelta` to delay a specified amount of time, or as an int to delay a specified number of ticks (delay must be positive)

## `csp.diff`

```python
csp.diff(
    x: ts['T'],
    lag: typing.Union[timedelta, int]
) → ts['T']
```

When `x` ticks, output difference between current tick and value time or ticks ago (once that exists)

## `csp.merge`

```python
csp.merge( x: ts['T'], y: ts['T']) → ts['T']
```

Merges the two timeseries `x` and `y` into a single series.
If both tick on the same cycle, the first input (`x`) wins and the value of `y` is dropped.
For loss-less merging see `csp.flatten`

## `csp.split`

```python
csp.split(
    flag: ts[bool],
    x: ts['T']
) → {false: ts['T'], true: ts['T']}
```

Splits input `x` onto two outputs depending on the value of `flag`.
If `flag` is `True` when `x` ticks, output 'true' will tick with the value of `x`.
If `flag` is `False` at the time of the input tick, then 'false' will tick.
Note that if flag is not valid at the time of the input tick, the input will be dropped.

## `csp.filter`

```python
csp.filter(flag: ts[bool], x: ts['T']) → ts['T']
```

Will only tick out input ticks of `x` if the current value of `flag` is `True`.
If flag is `False`, or if flag is not valid (hasn't ticked yet) then `x` is suppressed.

## `csp.drop_dups`

```python
csp.drop_dups(x: ts['T']) → ts['T']
```

Will drop consecutive duplicate values from the input.

## `csp.unroll`

```python
csp.unroll(x: ts[['T']]) → ts['T']
```

Given a timeseries of a *list* of values, unroll will "unroll" the values in the list into a timeseries of the elements.
`unroll` will ensure to preserve the order across all list ticks.
Ticks will be unrolled in subsequent engine cycles.
For a detailed explanation of this behavior, see the documentation on [duplicate timestamps](Execution-Modes#handling-duplicate-timestamps).

## `csp.collect`

```python
csp.collect(x: [ts['T']]) → ts[['T']]
```

Given a basket of inputs, return a timeseries of a *list* of all values that ticked

## `csp.flatten`

```python
csp.flatten(x: [ts['T']]) → ts['T']
```

Given a basket of inputs, return all ticks across all inputs as a single timeseries of type 'T'
(This is similar to `csp.merge` except that it can take more than two inputs, and is lossless)

## `csp.default`

```python
csp.default(
    x: ts['T'],
    default: '~T',
    delay: timedelta = timedelta()
)
```

Defaults the input series to the value of `default` at start of the engine, or after `delay` if `delay` is provided.
If `x` ticks right at the start of the engine, or before `delay` if `delay` is provided, `default` value will be discarded.

## `csp.gate`

```python
csp.gate(
    x: ts['T'],
    release: ts[bool]
) → ts[['T']]
```

`csp.gate` will hold values of the input series until `release` ticks `True`, at which point all pending values will be output as a burst.
`release` can tick open / closed repeatedly.
While open, the input will tick out as a single value burst.
While closed, input ticks will buffer up until they can be released.

## `csp.apply`

```python
csp.apply(
    x: csp.ts['T'],
    f: Callable[['T'], 'O'],
    result_type: 'O'
) → ts['O']
```

Applies the provided callable `f` on every tick of the input and returns the result of the callable.

## `csp.null_ts`

```python
csp.null_ts(typ: 'T')
```

Returns a "null" timeseries of the given type which will never tick.

## `csp.stop_engine`

```python
csp.stop_engine(x: ts['T'])
```

Forces the engine to stop if `x` ticks

## `csp.multiplex`

```python
csp.multiplex(
    x: {'K': ts['T']},
    key: ts['K'],
    tick_on_index: bool = False,
    raise_on_bad_key: bool = False
) → ts['T']
```

Given a dictionary basket of inputs and a key timeseries, tick out ticks from the input basket timeseries matching the current key.

Args:

- **`x`**: dictionary basket of timeseries inputs
- **`key`**: timeseries of keys that will be used as the multiplex key
- **`tick_on_index`**: if `True`, will tick the current value of
  the input basket whenever the key ticks (defaults to `False`)
- **`raise_on_bad_key`**: if `True` an exception will be raised if key ticks with an unrecognized key (defaults to `False`)

## `csp.demultiplex`

```python
csp.demultiplex(
    x: ts['T'],
    key: ts['K'],
    keys: ['K'],
    raise_on_bad_key: bool = False
) → {key: ts['T']}
```

Given a single timeseries input, a key timeseries to demultiplex on and a set of expected keys, will output the given input onto the corresponding basket output of the current value of `key`.
A good example use case of this is demultiplexing a timeseries of trades by account.
Assuming your trade struct has an account field, you can `demultiplex(trades, trades.account, [ 'acct1', 'acct2', ... ])`.

Args:

- **`x`**: the input timeseries to demultiplex
- **`key`**: a ticking timeseries of the current key to output to
- **`keys`**: a list of expected keys that will define the shape of the output basket.  The list of keys must be known at graph building time
- **`raise_on_bad_key`**: if `True` an exception will be raised of key ticks with an unrecognized key (defaults to `False`)

## `csp.dynamic_demultiplex`

```python
csp.dynamic_demultiplex(
    x: ts['T'],
    key: ts['K']
) → {ts['K']: ts['T']}
```

Similar to `csp.demultiplex`, this version will return a [Dynamic Basket](Create-Dynamic-Baskets) output that will dynamically add new keys as they are seen.

## `csp.dynamic_collect`

```python
csp.dynamic_collect(
    x: {ts['K']: ts['T']}
) → ts[{'K': 'T'}]
```

Similar to `csp.collect`, this function takes a [Dynamic Basket](Create-Dynamic-Baskets) input and returns a dictionary of the key-value pairs corresponding to the values that ticked.

## `csp.drop_nans`

```python
csp.drop_nans(x: ts[float]) → ts[float]
```

Filters nan (Not-a-number) values out of the time series.

## `csp.times`

```python
csp.times(x: ts['T']) → ts[datetime]
```

Given a timeseries, returns the time at which that series ticks

## `csp.times_ns`

```python
csp.times_ns(x: ts['T']) → ts[int]
```

Given a timeseries, returns the epoch time in nanoseconds at which that series ticks

## `csp.accum`

```python
csp.accum(x: ts["T"], start: "~T" = 0) -> ts["T"]
```

Given a timeseries, accumulate via `+=` with starting value `start`.

## `csp.exprtk`

```python
csp.exprtk(
    expression_str: str,
    inputs: {str: ts[object]},
    state_vars: dict = {},
    trigger: ts[object] = None,
    functions: dict = {},
    constants: dict = {},
    output_ndarray: bool = False)
```

Given a mathematical expression, and a set of timeseries corresponding to variables in that expression, tick out the result of that expression, either every time an input ticks, or on the trigger if provided.

Args:

- **`expression_str`**: an expression, as per the [C++ Mathematical Expression Library](http://www.partow.net/programming/exprtk/) (see [readme](http://www.partow.net/programming/exprtk/code/readme.txt))
- **`inputs`**: a dict basket of timeseries. The keys will correspond to the variables in the expression. The timeseries can be of float or string
- **`state_vars`**: an optional dictionary of variables to be held in state between executions, and assignable within the expression.  Keys are the variable names and values are the starting values
- **`trigger`**: an optional trigger for when to calculate. By default will calculate on any input tick
- **`functions`**: an optional dictionary whose keys are function names that can be used in the expression, and whose values are of the form `(("arg1", ..), "function body")`, for example `{"foo": (("x","y"), "x\*y")}`
- **`constants`**: an optional dictionary of constants.  Keys are constant names and values are their values
- **`output_ndarray`**: if `True`, output ndarray (1D) instead of float. Note that to output `ndarray`, the expression needs to use return like `return [a, b, c]`. The length of the array can vary between ticks.
