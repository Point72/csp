This page contains the documentation for the `csp.stats`  libary. The
`stats`  library contains functions to calculate statistics on time
series data over rolling windows.

## Table of Contents

1.  **Base Statistics**
    1. [count](#count): counts the number of data ticks within a given interval
    2. [unique](#unique): counts the number of unique values within a given interval
    3. [sum](#sum): rolling sum of values within a given interval
    4. [prod](#product): rolling product of values within a given interval
    5. [first](#first): the earliest value still within the interval
    6. [last](#last): the last value of the interval
    7. [mean](#mean): the mean of values within the interval
    8. [gmean](#geometric-mean): the geometric mean of values within the interval

2.  **Order Statistics**
    1. [max](#maximum): the maximum value within the interval
    2. [min](#minimum): the minimum value within the interval
    3. [median](#median): the median value within the interval
    4. [quantile](#quantile): the quantile value within the interval
    5. [argmin](#argmin): the time at which the minimum interval value ticked
    6. [argmax](#argmax): the time at which the maximum interval value ticked
    7. [rank](#rank): the time series rank of the most recent tick in the interval

3.  **Moment-Based Statistics**
    1. [var](#variance): variance of the time series within the interval
    2. [stddev](#standard-deviation): standard deviation within the interval
    3. [sem](#standard-error): standard error within the interval
    4. [cov](#covariance): covariance between two in-sequence time series within the interval
    5. [corr](#correlation): correlation between two in-sequence time series within the interval
    6. [skew](#skewness): skewness of the time series within the interval
    7. [kurt](#kurtosis): kurtosis (or excess kurtosis) of the time series within the interval

4.  **Exponential Moving Statistics**
    1. [ema](#exponential-moving-average): exponential moving average, with numerous different variations available
    2. [ema_var](#exponential-moving-variance): exponential moving variance
    3. [ema_std](#exponential-moving-standard-deviation): exponential moving standard deviation
    4. [ema_cov](#exponential-moving-covariance): exponential moving covariance between two in-sequence time series

5.  **NumPy Specific Statistics**
    1. [cov_matrix](#covariance-matrix): covariance matrix between *N* time-series (in a NumPy array) over a rolling time interval
    2. [corr_matrix](#correlation-matrix): normalized correlation matrix between *N* time-series (in a NumPy array) a rolling time interval
    3. [list_to_numpy](#numpy-conversions): converts a listbasket of time-series into a NumPy array
    4. [numpy_to_list](#numpy-conversions): converts a NumPy array time-series into a listbasket

6.  **Cross-Sectional Statistics**
    1.  [cross_sectional](#cross-sectional): receive all data within the current window for a cross-sectional calculation

## Base Statistics

### Count

```python
count(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data.
- **interval**: the rolling interval over which to use data.
    If unspecified or set to `None`, an expanding (unbounded) window will be used. 
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **ignore_na**: if `True`, ignores NaN values in the window (does not count them). If false, NaN values make the count NaN.
    - *By default*, `ignore_na` is True
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
    - *By default*, there is no reset series.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_points, NaN is returned.


Returns:
- A time-series of how many data points are currently in the interval. If a tick count is used, then it is necessarily less than or equal to the interval.


#### Examples: `count`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
count(x, interval=3)
```
```python
# NaN is not counted
{'2020-01-03': 3, '2020-01-04': 2, '2020-01-05': 2}
```

**2. Including NaN**

```python
count(x, interval=3, ignore_na=False)
```
```python
{'2020-01-03': 3, '2020-01-04': nan, '2020-01-05': nan}
```

**3. Triggering**
```python
trigger = {'2020-01-03': True, '2020-01-05': True}
count(x, interval=timedelta(days=3), min_window=timedelta(days=2), ignore_na=True, trigger=trigger)
```
```python
{'2020-01-03': 3, '2020-01-05': 2}
```

**4. Sampling**

```python
sampler = {'2020-01-01': True, '2020-01-02': True, '2020-01-03': True, '2020-01-05': True, '2020-01-06': True}
count(x, interval=timedelta(days=3), min_window=timedelta(days=2), sampler=sampler)
```
```python
{'2020-01-03': 3, '2020-01-05': 2}
```

**Note**: the x value at 2020-01-04 is ignored completely since sampler does not tick, while the value at 2020-01-06 is treated as NaN.

**5. Reset**
```python
reset = {'2020-01-04': True}
count(x, interval=timedelta(days=3), min_window=timedelta(days=2), reset=reset)
```
```python
{'2020-01-03': 3, '2020-01-04': 0, '2020-01-05': 1}
```
**Note**: the window data is reset at 2020-01-04, and its value is NaN, so the count is 0


**6. NumPy**
```python
x_np = {'2020-01-01': [1,1], '2020-01-02': [2,np.nan], '2020-01-03': [3,3]}
count(x_np, interval=3, min_window=1)
```
```python
{'2020-01-01': [1,1], '2020-01-02': [2,1], '2020-01-03': [3,2]} # count is per element
```



### Unique
```python
unique(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    precision: int = 10
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **trigger**: another optional time-series which can be use to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
    - *By default*, there is no reset series.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.
- **precision**: the decimal place precision at which two floats are considered non-unique. For example, if precision=2, then 2.001 and 2.002 would be considered non-unique.
    - *By default,* precision is set to 10 decimal places.

Returns:
- a time-series of how many unique (excluding nan) values are currently in the interval


#### Examples: `unique`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**
```python
x = {'2020-01-01': 2, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 3}
unique(x, interval=3, min_window=2)
```
```python
{'2020-01-02': 1, '2020-01-03': 2, '2020-01-04': 2, '2020-01-05': 1}
```

**2. Triggering**

```python
trigger = {'2020-01-03': True, '2020-01-05': True}
unique(x, interval=timedelta(days=3), min_window=timedelta(days=2), trigger=trigger)
```
```python
{'2020-01-03': 2, '2020-01-05': 1}
```

**3. NumPy**

```python
x_np = {'2020-01-01': [1,1], '2020-01-02': [2,np.nan], '2020-01-03': [3,1]}
unique(x_np, interval=3, min_window=1)
```
```python
{'2020-01-01': [1,1], 2020-01-02: [2,1], '2020-01-03': [3,1]} # unique is per element
```


### Sum
```python
sum(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    precise: bool = False,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data. Can either be a `ts[Union[float, np.ndarray]]` or `ts[np.ndarray]`.
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **precise**: if True we use a more numerically stable implementation (Kahan) which is less efficient
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values are included and will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **weights**: a time-series of weights for each observation in x, used to calculate a weighted sum (optional).
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**": another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling sums over the interval

####  Examples: `sum`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
sum(x, interval=3)
```
```python
{'2020-01-03': 6, '2020-01-04: 5', '2020-01-05': 8}
```

**2. Including NaNs**
```python
sum(x, interval=3, min_window=2, ignore_na=False)
```
```python
{'2020-01-02': 3, '2020-01-03': 6, '2020-01-04': nan, '2020-01-05': nan}
```

**3. Weighted single input**

```python
weights = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-04': 3}
sum(x_np, interval=3, weights=weights)
```
```python
{'2020-01-03': 11, '2020-01-04': 10, '2020-01-05': 21} # 21 = 5x3 + 3x2
```

**4. NumPy**
```python
x_np = {'2020-01-01': [1,1], '2020-01-02': [2,np.nan], '2020-01-03': [3,1]}
sum(x_np, interval=3, min_window=1)
```
```python
{'2020-01-01': [1,1], '2020-01-02': [3,1], '2020-01-03': [4,2]}
```

**5. NumPy weighted sum**

```python
np_weights = {'2020-01-01': [1,2], '2020-01-02': [2,1}
sum(x_np, interval=3, min_window=1, weights=np_weights)
```
```python
{'2020-01-01': [1,2], '2020-01-02': [5,2], '2020-01-03': [11,3]} # weights applied elementwise
```



### Product

```python
prod(
    x: ts[Union[float, np.ndarray]],
    interval : Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values are included and will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling products over the interval. The computation is unstable for large products and windows.


#### Examples: `prod`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**
```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
prod(x, interval=3, min_window=2, ignore_na=True)
```
```python
{'2020-01-02': 2, '2020-01-03': 6 '2020-01-04': 6, '2020-01-05': 15}
```

**2. NumPy**
```python
x_np = {'2020-01-01': [1,2], '2020-01-02': [3,4], '2020-01-03': [5,6]}
prod(x_np, 3, 2)
```
```python
{'2020-01-02': [3,8], '2020-01-03': [15,24]}
```

### First
```python
first(
    x: ts[Union[float, np.ndarray]],
    interval : Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    ignore_na: bool = True
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.
- **ignore_na**: if *True*, will return the first non-nan value in the window. If *False*, will return the first value in the window

Returns:
- a time-series of the earliest (non-nan) value still within the given interval


#### Examples: `first`
See `last`

### Last
```python
last(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data.
    If unspecified or set to None, an expanding (unbounded)
    window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep
        data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before
    outputting data
    - If the interval is a timedelta then this must also be a
        timedelta. Example: interval=60s, min_window=30s means
        to use a 60s rolling interval with no output for the
        first 30s.
    - If the interval is a tick count then this must also be a
        tick count. Example: interval=100, min_window=50 means
        to use a 100-tick rolling interval with no output until
        we have 50 ticks
    - <u>*By default,* the min_window is equal to the
        interval</u>
- **ignore_na**: if *True*, will return the last non-nan value
    in the window. If *False*, will return the last value in the
    window
- **trigger**: another optional time-series which can be used
    to externally trigger computations. Whenever the trigger
    ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of the most recent value within the given interval



#### Examples: `first` and `last`

Starttime: `2020-01-01 00:00:00`

**1. Default - first**
```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
first(x, interval=3)
```
```python
{'2020-01-03': 1, '2020-01-04': 2, '2020-01-05': 3}
```


**2. Including NaN - last**
```python
last(x, interval=3, ignore_na=False)
```
```python
{'2020-01-03': 3, '2020-01-04': nan, '2020-01-05': nan}
```

**3. Triggering - last**
```python
trigger = {'2020-01-03': True, '2020-01-04': True}
last(x, interval=timedelta(days=3), ignore_na=True, trigger=trigger)
```
```python
{'2020-01-03': 3, '2020-01-04': 3}
```

**4. NumPy - first**
```python
x_np = {'2020-01-01': [1,1], '2020-01-02': [2,np.nan], '2020-01-03': [3,3]}
first(x_np, interval=2)
```
```python
# first non-nan value
{'2020-01-02': [1,1], '2020-01-03': [2,3]}
```


### Mean

```python
mean(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```
- **x**: the time-series data. Can either be a `ts[Union[float, np.ndarray]]` or a `ts[np.ndarray]`.
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted mean (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling means over the interval. Computation uses smart updating so overflow is not an issue, since no sums are kept

#### Examples: `mean`
See `gmean`

### Geometric Mean
```python
gmean(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
)→ ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling geometric means over the interval. Requires a strictly positive-valued input.


#### Examples: `mean` and `gmean`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
mean(x, interval=3, min_window=2)
```

```python
{'2020-01-02': 1.5, '2020-01-03': 2.0, '2020-01-04': 2.5, '2020-01-05': 4.0}
```

**2. Including NaN**
```python
mean(x, interval=3, min_window=2, ignore_na=False)
```
```python
{'2020-01-02': 1.5, '2020-01-03': 2.0, '2020-01-04': nan, '2020-01-05': nan}
```

**3. Geometric mean**
```python
trigger = {'2020-01-03': True, '2020-01-05': True}
gmean(x, interval=timedelta(days=3), min_window=timedelta(days=2), ignore_na=True, trigger=trigger)
```
```python
{'2020-01-03': 1.817, '2020-01-05': 3.873}
```

**4. Weighted mean**

```python
weights = {'2020-01-01': 1, '2020-01-03': 2}
mean(x, interval=3, min_window=2, ignore_na=True, weights=weights)
```
```python
{'2020-01-02': 1.5, '2020-01-03': 2.25, '2020-01-04': 2.667, '2020-01-05': 4.0}
```
**Note**: the first two observations get relative weight of 1, then the last three get relative weight of 2

**5. NumPy weighted mean**
```python
x_np = {'2020-01-01': [1., 1., 1.], '2020-01-02': [2., 2., 2.], '2020-01-03': [3., 3., 3.]}
np_weights = {'2020-01-01': [1., 1., 1.], '2020-01-02': [2., 1., 2.], '2020-01-03': [3., 1., 3.]}
mean(x_np, 3, 2)
```
```python
{'2020-01-02': [1.667, 1.5, 1.667], '2020-01-03': [2.667, 2.0, 2.6667]}
```


## Order Statistics

### Maximum

```python
max(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
    - *By default,* the min_window is equal to the interval
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling maximums over the interval. 

#### Examples: `max`
See `min`

### Minimum

```python
min(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling minimums over the interval.


#### Examples: `max` and `min`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
min(x, interval=3, min_window=2)
```
```python
{'2020-01-02': 1, '2020-01-03': 1, '2020-01-04': 2, '2020-01-05': 3}
```

**2. Including NaN**
```python
max(x, interval=3, min_window=2, ignore_na=False)
```
```python
{'2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': nan}
```

**3. NumPy example**
```python
x_np= {'2020-01-01': [2,3], '2020-01-02': [6,1], '2020-01-03': [1,9]}
min(x, interval=timedelta(days=3), min_window=timedelta(days=1))
```
```python
{'2020-01-02': [2,1], '2020-01-03': [1,1]}
```


### Median
median(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling medians over the interval. Uses midpoint interpolation if there are an even number of samples.

#### Examples: `median`
See `quantile`


### Quantile
```
quantile(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    quant: Union[float, List[float]] = None,
    min_window: Union[timedelta, int] = None,
    interpolate: str = "linear",
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → Union[ts[Union[float, np.ndarray]], [ts[Union[float, np.ndarray]]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **quant:** the quantile to calculate, which must be between 0 and 1
    - If provided a list, then all quantiles will be calculated for the list. 
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks.
- **interpolate**: the interpolation method to use when the quantile does not correspond to an individual value. Must be one of the following options:
    - **"linear"**: interpolates linearly between the two closest values. For example, the 0.333 quantile of (1,2) with linear interpolation is 1.333.
    - **"lower"**: returns the lower of the two closest values.
    - **"higher"**: returns the higher of the two closest values.
    - **"midpoint"**: returns the midpoint between the two closest values. For example, the 0.333 quantile of (1,2) with midpoint interpolation is 1.5.
    - **"nearest"**: returns the value at the nearest position.  For example, the 0.333 quantile of (1,2) with nearest interpolation is 1. In cases of ties, the higher value is returned.
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series *or* list-basket of time-series of rolling quantiles over the interval. 
    - If the quant parameter is a list then a list-basket will be returned.
    - If it is a float then a time-series will be returned.
    - The order of quantiles in the list-basket is equal to the order of the input.


#### Examples: `median` and `quantile`

Starttime: `2020-01-01 00:00:00`

**1. Median**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
median(x, interval=3, min_window=2)
```
```python
{'2020-01-02': 1.5, '2020-01-03': 2, '2020-01-04': 2.5, '2020-01-05': 4}
```

**2. Quantile with multiple values**

```python
quantile(x, interval=3, quant=[0.25, 0.5, 0.75], min_window=2, ignore_na=False)
```
```python
[
    {'2020-01-02': 1.25, '2020-01-03': 1.5, '2020-01-04': nan, '2020-01-05': nan},
    {'2020-01-02': 1.5, '2020-01-03': 2.0, '2020-01-04': nan, '2020-01-05': nan},
    {'2020-01-02': 1.75, '2020-01-03': 2.5, '2020-01-04': nan, '2020-01-05': nan}
]
```

**3. Quantile with trigger**

```python
trigger = {'2020-01-03': True, '2020-01-05': True}
quantile(x, interval=timedelta(days=3), quant=0.333, min_window=timedelta(days=2), interpolate="midpoint", ignore_na=True, trigger=trigger)
```
```python
{'2020-01-03': 1.5, '2020-01-05': 4}
```

**4. NumPy array with multiple quantiles**

```python
x_np = {'2020-01-01': [1,2,3], '2020-01-02': [2,3,4], '2020-01-03': [3,4,5]}
quantile(x_np, interval=3, quant=[0.25,0.5,0.75], min_window=1)
```
```python
# this is a listbasket of NumPy array time series
[
    {'2020-01-01': [1,2,3], '2020-01-02': [1.25, 2.25, 3.25], '2020-01-03': [1.5, 2.5, 3.5]},
    {'2020-01-01': [1,2,3], '2020-01-02': [1.5, 2.5, 3.5], '2020-01-03': [2., 3., 4.]},
    {'2020-01-01': [1,2,3], '2020-01-02': [1.75, 2.75, 3.75], '2020-01-03': [2.5, 3.5, 4.5]}
]
```


### Argmin
```python
argmin(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    return_most_recent: bool = True,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[datetime, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **return_most_recent:** if True, in the case of a tie, the most recent time will be returned. If false, the least recent time will be returned. 
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. 

Returns:
- a time-series of rolling argmin values over the interval, returned as a datetime or NumPy array of np.datetime64 objects. If no data is present or NaN invalidation occurs, the default time '1970-1-1 00:00:00' is returned.


#### Examples: `argmin`
See `argmax`

### Argmax
```python
argmax(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    return_most_recent: bool = True,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[datetime, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **return_most_recent:** if True, in the case of a tie, the most recent time will be returned. If false, the least recent time will be returned. 
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling argmax values over the interval, returned as a datetime or NumPy array of np.datetime64 objects.  If no data is present or NaN invalidation occurs, the default time '1970-1-1 00:00:00' is returned.

####  Examples: `argmax and `argmin`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 1, '2020-01-04': nan, '2020-01-05': 4}
argmax(x, 3)
```
```python
{'2020-01-03': '2020-01-02', '2020-01-04': '2020-01-02', '2020-01-05': '2020-01-05'}
```

```python
argmin(x, 3)
```
```python
{'2020-01-03': '2020-01-03', '2020-01-04': '2020-01-03', '2020-01-05': '2020-01-03'}
```

**2. NumPy example**

```python
x_np = {'2020-01-01': [1,2], '2020-01-02': [2,1], '2020-01-03': [3,0]}
argmax(x_np, 3, 2)
```

```python
{'2020-01-02': ['2020-01-02', '2020-01-01'], '2020-01-03': ['2020-01-03', '2020-01-01']}
```

```python
argmin(x_np, 3, 1)
```
```python
{'2020-01-02': ['2020-01-01', '2020-01-02'], '2020-01-03': ['2020-01-01', '2020-01-03']}
```

**3. `return_most_recent=False`**

```python
argmin(x, 3, return_most_recent=False)
```
```python
{'2020-01-03': '2020-01-01', '2020-01-04': '2020-01-03', 2020-01-05: '2020-01-03'} # Note how the first element is '2020-01-01', not '2020-01-03'
```

### Rank
```python
rank(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    method: str = "min",
    ignore_na: bool = True,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    min_data_points: int = 0,
    na_option: str = "keep"
) → ts[Union[float, np.ndarray]]
```

Args:
  - **x**: the time-series data
  - **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
      - if an int, represents the number of ticks to use
      - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
  - **min_window**: the minimum allowable interval to use before outputting data
      - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
      - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use 100-tick rolling interval with no output until we have 50 ticks
  - **method**:  the method to use to rank groups of records
      that have the same value
      - **`"min"`**: the lowest rank in the group is returned i.e. if the window data is [1,2,2,3] and the last tick is 2, then rank=1
      - **`"max"`**: the highest rank in the group is returned i.e. if the window data is [1,2,2,3] and the last tick is 2, then rank=3
      - **`"avg"`**: the average rank in the group is returned i.e. if the window data is [1,2,2,3] and the last tick is 2, then rank=2
      - *By default,* the "min" method is used.
  - **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
  - **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
      - *By default*, the trigger is the series itself.
  - **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
      - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
      - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
      - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
      - *By default*, the sampler is the series itself.
  - **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
  - **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, nan is returned.
  - `na_option`: how to rank a nan value when it is the last
      value to be ranked
      - **`"keep"`**: return a nan rank for a nan value
      - **`"last"`**: rank the last non-nan value present in the interval
      - *By* default, the "keep" option is used.
- *Output*: a time-series of rolling ranks over the interval,
    where a rank of 0 means that the current (last) ticked value is
    the smallest in the given interval.

#### Examples: `rank`

Starttime: `2020-01-01 00:00:00`

**1. Default behavior**

```python
x = {'2020-01-01': 1, '2020-01-02': 3, '2020-01-03': 2, '2020-01-04': 5, '2020-01-05': 4}
rank(x, 5, min_window=3)
```
```python
{'2020-01-03': 1, '2020-01-04': 3, '2020-01-05': 3}
```

**2. NumPy example**

```python
x_np = {'2020-01-01': [1,2], '2020-01-02': [3,2], '2020-01-03': [2,1]}
rank(x_np, 3, 2)
```

```python
# Note how the second element at '2020-01-02' is 0, not 1, as by default the "min" method is used
{'2020-01-02': [1, 0], '2020-01-03': [1, 0]}
```

**3. "keep" vs "last" NaN option**

```python
x = {'2020-01-01': 1, '2020-01-02': 3, '2020-01-03': 2, '2020-01-04': nan, '2020-01-05': 4}
rank(x, 5, min_window=3, na_option="keep")
```
```python
{'2020-01-03': 1, '2020-01-04': nan, '2020-01-05': 3}
```
```python
rank(x, 5, min_window=3, na_option="last")
```
```python
# the last valid value, 1, is ranked at '2020-01-04'
{'2020-01-03': 1, '2020-01-04': 1, '2020-01-05': 3}
```

## Moment-Based Statistics

### Variance

```python
var(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ddof:** delta degrees of freedom. Example: if ddof=1, then normalization term is 1/(N-1). If ddof=0, then 1/N.
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted variance (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling variance over the interval. If insufficient samples for given ddof, then no value output is generated. Since the smart mean is being used, overflow is not a problem.

#### Examples: `var`
See Standard Error.

### Standard Deviation

```python
stddev(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ddof:** delta degrees of freedom
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted standard deviation (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling standard deviations over the interval. If insufficient samples for given ddof, then no value output is generated.

#### Examples: `stddev`
See Standard Error.

### Standard Error
```python
sem(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
): → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ddof:** delta degrees of freedom
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another optional time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned. 
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted standard error (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.


Returns:
- a time-series of rolling standard errors



#### Examples: Variance, Standard Deviation, Standard Error

Starttime: `2020-01-01 00:00:00`

**1. Variance**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
var(x, interval=3, min_window=2)
```
```python
{'2020-01-02': 0.5, '2020-01-03': 1.0, '2020-01-04': 0.5, '2020-01-05': 2.0}
```

**2. Biased variance**

```python
var(x, interval=3, min_window=2, ddof=0, ignore_na=True) # biased
```
```python
{'2020-01-02': 0.25, '2020-01-03': 0.666, '2020-01-04': 0.25, '2020-01-05': 1.0}
```

**3. Standard deviation including NaNs**

```python
stddev(x, interval=3, min_window=2, ignore_na=False)
```
```python
{'2020-01-02': 0.707, '2020-01-03': 1.0, '2020-01-04': nan, '2020-01-05': nan}
```

**4. Standard error with triggering**

```python
trigger = {'2020-01-03': True, '2020-01-05': True}
sem(x, interval=timedelta(days=3), min_window=timedelta(days=2), trigger=trigger)
```
```python
{'2020-01-03': 0.707, '2020-01-05': 1.0}
```



### Covariance
```python
cov(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
): → ts[Union[float, np.ndarray]]
```
Args:
- **x**: time-series data. If x is of type np.ndarray, then the covariance calculation is performed element-wise with the corresponding values in y.
- **y**: time-series data that ticks in sequence with x
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ddof:** delta degrees of freedom
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted covariance (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling covariances between x and y

#### Examples: `cov`
See Correlation.

### Correlation

```python
corr(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
): → ts[Union[float, np.ndarray]]
```

Args:
- **x**: time-series data. If x is of type np.ndarray, then the correlation calculation is performed element-wise with the corresponding values in y.
- **y**: time-series data that ticks in sequence with x
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted correlation (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling Pearson correlation coefficients between x and y


#### Examples: Covariance and Correlation

Starttime: `2020-01-01 00:00:00`

**1. Covariance**
```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': 4, '2020-01-05': 5}
y = {'2020-01-01': 5, '2020-01-02': 4, '2020-01-03': 3, '2020-01-04': 2, '2020-01-05': 1}
cov(x, y, interval=3, min_window=2)
```
```python
{'2020-01-02': -0.5, '2020-01-03': -1.0, '2020-01-04': -1.0, '2020-01-05': -1.0}
```

**2. Correlation**

```python
corr(x, y, interval=3)
```
```python
{'2020-01-03': -1.0, '2020-01-04': -1.0, '2020-01-05': -1.0}
```

### Skewness
```
skew(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    bias: bool = False,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
): → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **bias:** if True, calculates a biased (unadjusted) skew. If false (default), calculates a Gaussian-unbiased measure.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted skew (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling sample skew measures, using the adjusted Fisher–Pearson standardized moment coefficient.

#### Examples: `skew`
See Kurtosis.

### Kurtosis
```python
kurt(
    x: ts[Union[float, np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    excess: bool = True,
    bias: bool = False,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
): → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data.
    If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **excess:** if True (default) uses the definition of excess kurtosis (kurt - 3). If false, uses the standard definition.
- **bias:** if True, calculates a biased (unadjusted) kurtosis. If false (default), calculates a Gaussian-unbiased measure.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted kurtosis (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the window statistic, and in doing so clears any accumulated floating-point error
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of rolling sample kurtosis measures, using the adjusted Fisher–Pearson standardized moment coefficient.

#### Examples: `skew` and `kurt`

Starttime: `2020-01-01 00:00:00`

**1. Skew**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, ..., 2020-01-10: 10}
skew(x, interval=7)
```

```python
{2020-01-07: 0, 2020-01-08: 0, 2020-01-09: 0, 2020-01-10: 0}
```

**2. Kurtosis**

```python
kurt(x, interval=7) # excess kurtosis
```

```python
{2020-01-07: -1.2, 2020-01-08: -1.2, 2020-01-09: -1.2, 2020-01-10: -1.2}
```

## Exponential Moving Statistics

### Exponential Moving Average
```python
ema(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: Optional[timedelta] = None,
    adjust: bool = True,
    horizon: int = None,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **min_periods**: the minimum allowable number of ticks to use before outputting data. The default is 1 for any EMA function.
- **alpha**: the EMA weight parameter specified directly.
    If *adjust = True,* EMA is calculated such that 

    $$EMA(t) = \frac{\sum\limits_{t=-n}^{0} (1-\alpha)^{-t} x(-t)}{\sum\limits_{t=-n}^{0} (1-\alpha)^{-t}}$$

    If `adjust = False`, EMA is calculated such that

    $$EMA(t) = (1-\alpha)EMA(t-1) + \alpha x(t)$$
    $$EMA(t=0) = x(0)$$

    By default, adjust = True, to give better estimates for starting intervals. 

    The following are alternative methods to specify the $\alpha$ parameter.

  - **span**: specify alpha in terms of span, such that 

    $$\alpha = \frac{2}{span+1}$$

  - **com**: specify alpha in terms of centre of mass, such that 

    $$\alpha = \frac{1}{1+com}$$

  - **halflife**: Halflife is different from the other parameters. Half-life is a timedelta argument that specifies the half-life of observation weights. Half-life is useful when observations are irregularly spaced and a better estimate is needed to properly weight more recent data. Let $t_{-1}$ be the time of the last observation.

    Then:

    $$\lambda(t)  = 1 - \exp(\frac{-(t-t_{-1})*\ln(2)}{halflife})$$
    $$EMA(t) = \frac{ \lambda(t)*EMA(t-1) + x(t)}{\text{normalization constant}}$$

    Something to note is that the `ignore_na` flag does not matter if a halflife interval is specified.
    The behavior would be the same in both cases, since an absolute time interval is being used to re-weight the moving average, not a tick interval.

    **Exactly one of alpha, span, com, halflife must be given**

- **adjust**: if True, early observations are adjusted to give a more "smoothed" estimate of the EMA. The difference is that if `adjust=True`, then each new observation receives a relative weight of 1. If adjust = False, each new observation receives a relative weight of alpha. 
  - `adjust=True` means that:

  $$EMA(t) = \frac{x(t)+(1-\alpha)x(t-1)+(1-\alpha)^2 x(t-2) + ... + (1-\alpha)^n x(t-n)}{1+(1-\alpha)+(1-\alpha)^ 2 + ... + (1-\alpha)^n}$$

  - `adjust=False` means that:

  $$EMA(t) = \frac{\alpha * x(t) + \alpha * (1-\alpha) * x(t-1) + \alpha * (1-\alpha)^2 * x(t-2) + ... + \boldsymbol{(1-\alpha)^n x(0)}}{\alpha+\alpha*(1-\alpha)+\alpha*(1-\alpha)^ 2 + ... + (1-\alpha)^n}$$

  $$\text{and thus } EMA(t=0) = x(0)$$

    Adjust only applies with tick specified intervals, not time specified intervals. Time specified intervals (i.e. half-life) do not need adjustment as they are, by definition, already adjusted.
- **horizon**: the maximum number of ticks to use in the computation. For example, if horizon = 10, then only the 10 most recent data points are used. If not specified, all data points for x are used, with early ticks decaying exponentially in weighting. Horizon will be ignored with a half-life (time-based) interval.
    - If horizon is set to *h*, then even if x has more than *h* ticks the EMA will computed as such if `adjust=True`.

    $$EMA(t) = \frac{\sum_{t=-h}^{0} (1-\alpha)^{-t} x(t)}{\sum_{t=-h}^{0} (1-\alpha)^{-t}}$$

    -  The only difference if `adjust=False` is that the first ever tick, while in the window, receives weight 1 at the start instead of weight  $\alpha$ like the rest of the values.

- **ignore_na**: if True, nan values will be "ignored" meaning weights will be placed on relative position. If False (default), weights are based on global position, and renormalized as such.
  - For example, let us consider a dataset (1,nan,2) using `adjust=True`.
    - If `ignore_na=True` then the weighting is based on *relative position* as such:
    $$EMA(t=2) = \frac{(1-\alpha)*1 + 2}{(1-\alpha)+1}$$
    - If `ignore_na=False` then the weighting is based on *global position* as such:
    $$EMA(t=2) = \frac{(1-\alpha)^2*1 + 2}{(1-\alpha)^2+1}$$
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation.
- **recalc**: another optional time-series which triggers a clean recalculation of the EMA, and in doing so clears any accumulated floating-point error. 
    - Note: *only valid when a finite-horizon EMA is used*.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of exponentially-weighted moving averages over the interval.

####  Examples: `ema`

Starttime: `2020-01-01 00:00:00`

**1. Unadjusted EMA**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': 4, '2020-01-05': 5}
ema(x, alpha=0.1, adjust=False) # unadjusted
```

```python
{'2020-01-01': 1.0, '2020-01-02': 1.1, '2020-01-03': 1.29, '2020-01-04': 1.561, '2020-01-05': 1.9049}
```

**2. Adjusted EMA**

```python
ema(x, alpha=0.1, adjust=True)  # adjusted, default method
```
```python
{'2020-01-01': 1.0, '2020-01-02': 1.5263, '2020-01-03': 2.0701, '2020-01-04': 2.6313, '2020-01-05': 3.20971}
```

**3. Finite horizon EMA**
```python
ema(x, alpha=0.1, adjust=True, horizon=2) # finite horizon
```

```python
{'2020-01-01': 1.0, '2020-01-02': 1.5263, '2020-01-03': 2.5263, '2020-01-04': 3.5263, '2020-01-05': 4.5263}
```

**4. Time-based decay EMA**

```python
ema(x, halflife=timedelta(days=1)) # time-based
```
```python
{'2020-01-01': 1.0, '2020-01-02': 1.6666, '2020-01-03': 2.4286, '2020-01-04': 3.2666, '2020-01-05': 4.1613}
```

**5. Unadjusted EMA for NumPy array**

```python
x_np = {'2020-01-01': [1,2], '2020-01-02': [4,5], '2020-01-03': [7,8]}
ema(x_np, alpha=0.1, adjust=False)
```
```python
{'2020-01-01': [1,2], '2020-01-02': [1.3,2.3], '2020-01-03': [1.87,2.87] }
```

### Exponential Moving Variance
```python
ema_var(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: Optional[Union[float, timedelta]] = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```
Args:
- **x**: the time-series data
- **min_periods**: the minimum allowable number of ticks to use before outputting data. The default is 1 for any EMA function.
- **alpha,** **span, com, halflife**: as described in EMA
- **adjust**: as specified in EMA
- **horizon**: as specified in EMA. 
- **bias:** if True, uses a biased population weighted variance. If false, normalized by a proper debiasing factor.
- **ignore_na**: if True, nan values will be "ignored" meaning weights will be placed on relative position. If False (default), weights are based on global position.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the EMA, and in doing so clears any accumulated floating-point error. 
    - Note: *only valid when a finite-horizon EMA is used*.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of exponentially-weighted moving variances over the interval. 

#### Examples: `ema_var`
See Exponential Moving Standard Deviation

### Exponential Moving Standard Deviation
```
ema_std(
    x: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: Optional[Union[float, timedelta]] = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0,
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: the time-series data
- **min_periods**: the minimum allowable number of ticks to use before outputting data. The default is 1 for any EMA function.
- **alpha,** **span, com, halflife**: as described in EMA
- **adjust**: as specified in EMA
- **horizon**: as specified in EMA. 
- **bias:** if True, uses a biased population weighted variance. If false, normalized by debiasing factor
- **ignore_na**: if True, nan values will be "ignored" meaning weights will be placed on relative position. If False (default), weights are based on global position.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the EMA, and in doing so clears any accumulated floating-point error. 
    - Note: *only valid when a finite-horizon EMA is used*.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of exponentially-weighted moving standard deviations over the interval. 

#### Examples: Exp. Moving Variance and Standard Deviation

Starttime: `2020-01-01 00:00:00`

**1. Exp. Moving Standard Deviation**

```python
x = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3, '2020-01-04': nan, '2020-01-05': 5}
ema_std(x, min_periods=2, span=20, adjust=False, bias=False, ignore_na=False)
```

```python
{'2020-01-02': 0.707, '2020-01-03': 1.11636, '2020-01-04': 1.11636, '2020-01-05': 1.937005}
```

**2. Exp. Moving Variance**

```python
ema_var(x, min_periods=2, span=20, adjust=False, bias=True, ignore_na=False)
```
```python
{'2020-01-02': 0.086168, '2020-01-03': 0.390588 '2020-01-04': 0.390588, '2020-01-05': 1.644124}
```

### Exponential Moving Covariance
```python
ema_cov(
    x: ts[Union[float, np.ndarray]],
    y: ts[Union[float, np.ndarray]],
    min_periods: int = 1,
    alpha: Optional[float] = None,
    span: Optional[float] = None,
    com: Optional[float] = None,
    halflife: Optional[Union[float, timedelta]] = None,
    adjust: bool = True,
    horizon: int = None,
    bias: bool = False,
    ignore_na: bool = False,
    trigger: ts[object] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[Union[float, np.ndarray]]
```

Args:
- **x**: time-series data. If x is of type np.ndarray, the exponential-moving covariance is calculated element-wise with the corresponding values in y.
- **y:** time-series data which ticks in-sequence with x
- **min_periods**: the minimum allowable number of ticks to use before outputting data. The default is 1 for any EMA function.
- **alpha,** **span, com, halflife**: as described in EMA
- **adjust**: as specified in EMA
- **horizon**: as specified in EMA. 
- **bias:** if True, uses a biased population weighted covariance. If false, normalized by debiasing factor
- **ignore_na**: if True, nan values will be "ignored" meaning
    weights will be placed on relative position. If False (default), weights are based on global position.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the EMA, and in doing so clears any accumulated floating-point error. 
    - Note: *only valid when a finite-horizon EMA is used*.
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of exponentially-weighted moving covariance over the interval.

## NumPy Specific Statistics

### Covariance Matrix

```python
cov_matrix(
    x: ts[np.ndarray],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ddof: int = 1,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[np.ndarray]
```

Args:
- **x**: the time-series of dimension `(N,)` arrays which represent `N` variables
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ddof:** delta degrees of freedom
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted covariance matrix (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the statistic, and in doing so clears any accumulated floating-point error. 
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of (potentially weighted) covariance matrices, each of which is a NumpyNDArray of dimensionality `(N,N)`

### Correlation Matrix

```python
corr_matrix(
    x: ts[np.ndarray],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    ignore_na: bool = True,
    trigger: ts[object] = None,
    weights: ts[Union[float, np.ndarray]] = None,
    sampler: ts[object] = None,
    reset: ts[object] = None,
    recalc: ts[object] = None,
    min_data_points: int = 0
) → ts[np.ndarray]
```
Args:
- **x**: the time-series of dimension `(N,)` arrays which represent `N` variables
- **interval**: the rolling interval over which to use data.
    If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **ignore_na**: if True, does not include any nan values in the window. If false, nan values in the window will make the entire window value nan.
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **weights:** a time-series of weights for each observation in x, used to calculate a weighted correlation matrix (optional). Weights do not need to be normalized.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation
- **recalc**: another optional time-series which triggers a clean recalculation of the statistic, and in doing so clears any accumulated floating-point error. 
- **min_data_points**: the minimum number of valid (non-nan) data points that must exist in the interval for a calculation to be valid. If there are fewer than min_data_point, NaN is returned.

Returns:
- a time-series of (potentially weighted) correlation matrices, each of which is a NumpyNDArray of dimensionality `(N,N)`

#### Examples: Covariance and Correlation Matrices

Starttime: `2020-01-01 00:00:00`

**1. Covariance**

```python
x = {'2020-01-01': np.array([0., 0., 0.]), '2020-01-02': np.array([1., -1., 2.]), '2020-01-03': np.array([2., -2., 4.])}
cov_matrix(x, 3, ddof=0)
```

```python
{'2020-01-03': np.array([1, -1, 2],
                     [-1, 1, -2],
                      [2, -2, 4])}
```

**2. Correlation**

```python
corr_matrix(x, 3)
```

```python
{'2020-01-03': np.array([1, -1, 1],
                     [-1, 1, -1],
                      [1, -1, 1])}
```

### NumPy Conversions

```python
list_to_numpy(x: [ts[float]], fillna: bool = False) → ts[np.ndarray]
```
Args:
- **x**: a listbasket of time series 
- **fillna**: If False, unticked elements are treated as NaN.
    If True, unticked elements will hold their previous value in the array. 

Returns:
- a NumPy 1D array where each value corresponds to the element of the listbasket with the same index

```python
numpy_to_list(x: ts[np.ndarray], n: int) → [ts[float]]
```
Args: 
- **x**: a NumPy array valued time series 
- **n**: the number of output channels in the listbasket
Returns:
- a listbasket where each value corresponds to the element of the array with the same index

#### Examples: NumPy Conversions

Starttime: `2020-01-01 00:00:00`

**1. List to NumPy**

```python
x1 = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3}
x2 = {'2020-01-01': 1.5, '2020-01-03': 3.5}
list_to_numpy([x1,x2], fillna=False)
```
```python
{'2020-01-01': [1, 1.5], '2020-01-02': [2, np.nan], '2020-01-03': [3, 3.5]} # no x2 tick on day 2
```
```python
list_to_numpy([x1,x2], fillna=True)
```
```python
{'2020-01-01': [1, 1.5], '2020-01-02': [2, 1.5], '2020-01-03': [3, 3.5]} # holds x2 value for day 2
```

**2. NumPy to list**

```python
x_np = {'2020-01-01': [1,2], '2020-01-02': [3,4], '2020-01-03': [5,6]}
numpy_to_list(x_np, 2)
```
```python
[
    {'2020-01-01': 1, '2020-01-02': 3, '2020-01-03': 5},
    {'2020-01-01': 2, '2020-01-02': 4, '2020-01-03': 6}
]
```

## Cross-Sectional Statistics

### Cross Sectional
```python
cross_sectional(
    x: ts[Union[float,np.ndarray]],
    interval: Union[timedelta, int] = None,
    min_window: Union[timedelta, int] = None,
    trigger: ts[object] = None,
    as_numpy: bool = False,
    sampler: ts[object] = None,
    reset: ts[object] = None
) → ts[Union[np.ndarray, List[float], List[np.ndarray]]]
```

Args:
- **x**: the time-series data
- **interval**: the rolling interval over which to use data. If unspecified or set to None, an expanding (unbounded) window will be used.
    - if an int, represents the number of ticks to use
    - if a timedelta, represents the time interval to keep data (non-inclusive at left endpoint)
- **min_window**: the minimum allowable interval to use before outputting data
    - If the interval is a timedelta then this must also be a timedelta. Example: interval=60s, min_window=30s means to use a 60s rolling interval with no output for the first 30s.
    - If the interval is a tick count then this must also be a tick count. Example: interval=100, min_window=50 means to use a 100-tick rolling interval with no output until we have 50 ticks
- **as_numpy:** if True, the data will be returned as a NumPy array instead of a list.
    - For a single-valued time series, this is a one-dimensional NumPy array
    - For a NumPy array time series, this is a NumPy array of one extra dimension
- **trigger**: another time-series which can be used to externally trigger computations. Whenever the trigger ticks, the given statistic will be updated and returned
    - *By default*, the trigger is the series itself.
- **sampler**: another optional time-series which specifies when x *should* tick. The behavior is as follows:
    - If x ticks *and *sampler ticks, then the x tick is considered valid and is used.
    - If x ticks but sampler does not tick, then the x tick is considered invalid and is ignored.
    - If x does not tick but sampler ticks, then the x tick is considered NaN and is handled based on the ignore_na flag.
    - *By default*, the sampler is the series itself.
- **reset**: another optional time-series which, when ticked, will clear all data in the interval and "reset" the calculation

Returns:
- a time-series where each tick contains all the data of *x* currently within the interval. Use this for custom cross-sectional calculations

#### Examples: Cross-sectional calculations

Starttime: `2020-01-01 00:00:00`

```python
x = {'2020-01-01': 1, '2020-01-01': 2, '2020-01-01': 3, '2020-01-01': 4, '2020-01-01': 5}
cs = cross_sectional(x, interval=3, min_window=2)
cs
```
```python
{'2020-01-02': [1,2], '2020-01-03': [1,2,3], '2020-01-04': [2,3,4], '2020-01-05': [3,4,5]}
```

**Calculate a cross-sectional mean**

```python
cs_mean = csp.apply(cs, lambda v: sum(v)/len(v), float)
cs_mean
```

```python
{'2020-01-02': 1.5, '2020-01-03': 2.0, '2020-01-04': 3.0, '2020-01-05': 4.0}
```

**Get the results as a NumPy array**

```python
cs = cross_sectional(x, interval=3, min_window=2, as_numpy=True)
cs
```

```python
{'2020-01-02': np.array([1,2]), '2020-01-03': np.array([1,2,3]), '2020-01-04': np.array([2,3,4]), '2020-01-05': np.array([3,4,5])}
```
