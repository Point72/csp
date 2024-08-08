## Table of Contents

- [Table of Contents](#table-of-contents)
- [Historical Buffers](#historical-buffers)
- [Historical Range Access](#historical-range-access)

## Historical Buffers

CSP can provide access to historical input data as well.
By default only the last value of an input is kept in memory, however one can request history to be kept on an input either by number of ticks or by time using **csp.set_buffering_policy.**

The methods **csp.value_at**, **csp.time_at** and **csp.item_at** can be used to retrieve historical input values.
Each node should call **csp.set_buffering_policy** to make sure that its inputs are configured to store sufficiently long history for correct implementation.
For example, let's assume that we have a stream of data and we want to create equally sized buckets from the data.
A possible implementation of such a node would be:

```python
@csp.node
def data_bin_generator(bin_size: int, input: ts['T']) -> ts[['T']]:
    with csp.start():
        assert bin_size > 0
        # This makes sure that input stores at least bin_size entries
        csp.set_buffering_policy(input, tick_count=bin_size)
    if csp.ticked(input) and (csp.num_ticks(input) % bin_size == 0):
        return [csp.value_at(input, -i) for i in range(bin_size)]
```

In this example, we use **`csp.set_buffering_policy(input, tick_count=bin_size)`** to ensure that the buffer history contains at least **`bin_size`** elements.
Note that an input can be shared by multiple nodes, if multiple nodes provide size requirements, the buffer size would be resolved to the maximum size to support all requests.

Alternatively, **`csp.set_buffering_policy`** supports a **`timedelta`** parameter **`tick_history`** instead of **`tick_count`.**
If **`tick_history`** is provided, the buffer will scale dynamically to ensure that any period of length **`tick_history`** will fit into the history buffer.

To identify when there are enough samples to construct a bin we use **`csp.num_ticks(input) % bin_size == 0`**.
The function **`csp.num_ticks`** returns the number or total ticks for a given time series.
NOTE: The actual size of the history buffer is usually less than **`csp.num_ticks`** as buffer is dynamically truncated to satisfy the set policy.

The past values in this example are accessed using **`csp.value_at`**.
The various historical access methods take the same arguments and return the value, time and tuple of `(time,value)` respectively:

- **`csp.value_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns **value** of the timeseries at requested `index_or_time`
- **`csp.time_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns **datetime** of the timeseries at requested `index_or_time`
- **`csp.item_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns tuple of `(datetime,value)` of the timeseries at requested `index_or_time`
  - **`ts`**: the name of the input
  - **`index_or_time`**:
    - If providing an **index**, this represents how many ticks back to rereieve **and should be \<= 0**.
      0 indicates the current value, -1 is the previous value, etc.
    - If providing **time** one can either provide a datetime for absolute time, or a timedelta for how far back to access.
      **NOTE** that timedelta must be negative to represent time in the past..
  - **`duplicate_policy`**: when requesting history by datetime or timedelta, its possible that there could be multiple values that match the given time.
    **`duplicate_policy`** can be provided to control the behavior of what to return in this case.
    The default policy is to return the LAST_VALUE that exists at the given time.
  - **`default`**: value to be returned if the requested time is out of the history bounds (if default is not provided and a request is out of bounds an exception will be raised).

The following demonstrate a possible way to compute a rolling sum for the past N ticks.  Please note that this is for demonstration purposes only and is not efficient.  A more efficient
vectorized version can be seen below, though even that would not be recommended for a rolling sum since csp.stats.sum would be even more efficient with its C++ impl in-line calculation

```python
@csp.node
def rolling_sum(x:ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        csp.set_buffering_policy(x, tick_count=tick_count)

    if csp.ticked(x):
        return sum(csp.value_at(x, -i) for i in range(min(csp.num_ticks(x), tick_count)))
```

## Historical Range Access

In similar fashion, the methods **`csp.values_at`**, **`csp.times_at`** and **`csp.items_at`** can be used to retrieve a range of historical input values as numpy arrays.
The sample_sum example above can be accomplished more efficiently with range access:

```python
@csp.node
def rolling_sum(x:ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        csp.set_buffering_policy(x, tick_count=tick_count)

    if csp.ticked(x):
        return csp.values_at(x).sum()
```

The past values in this example are accessed using **`csp.values_at`**.
The various historical access methods take the same arguments and return the value, time and tuple of `(times,values)` respectively:

- **`csp.values_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns values in specified range as a numpy array
- **`csp.times_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns times in specified range as a numpy array
- **`csp.items_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns a tuple of (times, values) numpy arrays
  - **`ts`** - the name of the input
  - **`start_index_or_time`**:
    - If providing an **index**, this represents how many ticks back to retrieve **and should be \<= 0**.
      0 indicates the current value, -1 is the previous value, etc.
    - If providing  **time** one can either provide a datetime for absolute time, or a timedelta for how far back to access.
      **NOTE that timedelta must be negative** to represent time in the past..
    - If **None** is provided, the range will begin "from the beginning" - i.e., the oldest tick in the buffer.
  - **`end_index_or_time`**: same as start_index_or_time
    - If **None** is provided, the range will go "until the end" - i.e., the newest tick in the buffer.
  - **`start_index_policy`**: only for use with datetime/timedelta as the start and end parameters.
    - **`TimeIndexPolicy.INCLUSIVE`**: if there is a tick exactly at the requested time, include it
    - **`TimeIndexPolicy.EXCLUSIVE`**: if there is a tick exactly at the requested time, exclude it
    - **`TimeIndexPolicy.EXTRAPOLATE`**: if there is a tick at the beginning timestamp, include it.
      Otherwise, if there is a tick before the beginning timestamp, force a tick at the beginning timestamp with the prevailing value at the time.
  - **`end_index_policy`**: only for use with datetime/timedelta and the start and end parameters.
    - **`TimeIndexPolicy.INCLUSIVE`**: if there is a tick exactly at the requested time, include it
    - **`TimeIndexPolicy.EXCLUSIVE`**: if there is a tick exactly at the requested time, exclude it
    - **`TimeIndexPolicy.EXTRAPOLATE`**: if there is a tick at the end timestamp, include it.
      Otherwise, if there is a tick before the end timestamp, force a tick at the end timestamp with the prevailing value at the time

Range access is optimized at the C++ layer and for this reason its far more efficient than calling the single value access methods in a loop, and they should be substituted in where possible.

Below is a rolling average example to illustrate the use of timedelta indexing.
Note that `timedelta(seconds=-n_seconds)` is equivalent to `csp.now() - timedelta(seconds=n_seconds)`, since datetime indexing is supported.

```python
@csp.node
def rolling_average(x: ts[float], n_seconds: int) -> ts[float]:
    with csp.start():
        assert n_seconds > 0
        csp.set_buffering_policy(x, tick_history=timedelta(seconds=n_seconds))
    if csp.ticked(x):
        avg = np.mean(csp.values_at(x, timedelta(seconds=-n_seconds), timedelta(seconds=0),
                                      csp.TimeIndexPolicy.INCLUSIVE, csp.TimeIndexPolicy.INCLUSIVE))
        csp.output(avg)
```

When accessing all elements within the buffering policy window like
this, it would be more succinct to pass None as the start and end time,
but datetime/timedelta allows for more general use (e.g. rolling average
between 5 seconds and 1 second ago, or average specifically between
9:30:00 and 10:00:00)
