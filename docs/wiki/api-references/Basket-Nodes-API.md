The `csp.basketlib` library contains useful functions for working with list and dict baskets.
These functions are found in the `csp.basketlib` module and can be called using `csp.basketlib.<func>`.

## Table of Contents

- [`sync_list`](#sync_list)
- [`sync_dict`](#sync_dict)
- [`sync`](#sync)
- [`sample_list`](#sample_list)
- [`sample_dict`](#sample_dict)
- [`sample_basket`](#sample_basket)

## `sync_list`

```python
sync_list(x: List[ts["T"]], threshold: timedelta, output_incomplete: bool = True) → csp.OutputBasket(
    List[ts["T"]], shape_of="x"
)
```

Synchronizes a list basket of time series within some threshold.

When any element of `x` first ticks, we wait up to `threshold` time for other elements to tick. Once all elements of the list basket tick at least once *or* the threshold elapses and `output_incomplete=True`, we return a list basket with the most recent value of each time series (between the interval's first tick and now) and reset the synchronization interval.

Args:

- **`x`**: a list basket of time series to synchronize.
- **`threshold`**: the time to wait for all elements of the basket to tick before propagating the values.
- **`output_incomplete`**: if True, return an incomplete output basket if the threshold elapses before all values tick. Else, do not output in this situation.

## `sync_dict`

```python
sync_dict(x: Dict["K", ts["T"]], threshold: timedelta, output_incomplete: bool = True) → csp.OutputBasket(
    Dict["K", ts["T"]], shape_of="x"
)
```

Synchronizes a dict basket of time series within some threshold. For the specific synchronization behavior, see the docs for `sync_list` above.

Args:

- **`x`**: a dict basket of time series to synchronize.
- **`threshold`**: the time to wait for all elements of the basket to tick before propagating the values.
- **`output_incomplete`**: if True, return an incomplete output basket if the threshold elapses before all values tick. Else, do not output in this situation.

## `sync`

```python
sync(x, threshold: timedelta, output_incomplete: bool = True)
```

Helper function which calls `sync_list` if x is a list basket and `sync_dict` if x is a dict basket. If x is not a valid basket, it will raise an exception.

## `sample_list`

```python
sample_list(trigger: ts["Y"], x: List[ts["T"]]) → csp.OutputBasket(List[ts["T"]], shape_of="x")
```

Samples a list basket of time series on a common trigger.

Args:

- **`trigger`**: when trigger ticks, sample the most recent value of each element in `x`.
- **`x`**: the list basket of time series to sample.

## `sample_dict`

```python
sample_dict(trigger: ts["Y"], x: Dict["K", ts["T"]]) → csp.OutputBasket(Dict["K", ts["T"]], shape_of="x")
```

Samples a dict basket of time series on a common trigger.

Args:

- **`trigger`**: when trigger ticks, sample the most recent value of each element in `x`.
- **`x`**: the dict basket of time series to sample.

## `sample_basket`

```python
sample_basket(trigger, x)
```

Helper function which calls `sample_list` if x is a list basket and `sample_dict` if x is a dict basket. If x is not a valid basket, it will raise an exception.
