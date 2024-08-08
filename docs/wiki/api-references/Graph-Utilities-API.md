The main utility functions of CSP are documented here. These functions enable users to execute and visualize graphs.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [`csp.run`](#csprun)
- [`csp.run_on_thread`](#csprun_on_thread)
- [`csp.show_graph`](#cspshow_graph)

## `csp.run`

```python
csp.run(
    g,
    *args,
    starttime=None,
    endtime=MAX_END_TIME,
    realtime=False,
    output_numpy=False,
    **kwargs,
):
```

Runs the graph `g` from `starttime` to `endtime`.

Args:

- **`g`**: a graph function or CSP node.
- **`args`**: any positional arguments passed to `g`.
- **`starttime`**: the engine time that the graph will start at.
- **`endtime`**: the engine time that the graph will end at. Can be a `datetime` or a `timedelta`. If a `timedelta`, the end time is treated as an offset from the start time.
- **`realtime`**: whether the graph runs in realtime versus historical mode. Defaults to historical.
- **`output_numpy`**: if True, will return each output of `g` as a two element tuple of Numpy arrays, one for the times and one for the values.
- **`kwargs`**: any keyword arguments passed to `g`.

Returns the outputs of `g` in a dictionary where the key is the output name and the value is a list of tuples. Each tuple has two elements, the time of the event and its value.

## `csp.run_on_thread`

```python
def run_on_thread(
    g,
    *args,
    starttime=None,
    endtime=MAX_END_TIME,
    queue_wait_time=None,
    realtime=False,
    auto_shutdown=False,
    daemon=False,
    **kwargs,
):
```

Runs the graph `g` on a separate thread.

Args:

See `csp.run` for overlapping arguments.

- **`auto_shutdown`**: if True, the graph will automatically stop and join when the runner goes out scope. Defaults to False.
- **`daemon`**: if True, will launch the thread as a daemon. Defaults to False.

Returns a runner object, which has the following methods:

```python
    def join(self, suppress=False):
        """wait for engine thread to finish and return results. If suppress=True, will suppress exceptions."""

    def is_alive(self):
        """Checks whether the thread is still running"""

    def stop_engine(self):
        """request engine to stop ( async )"""
```

## `csp.show_graph`

```python
def show_graph(
  graph_func, 
  *args, 
  graph_filename=None, 
  **kwargs,
):
```

Displays the graph using `graphviz`.

Args:

- **`graph_func`**: a graph function or CSP node.
- **`args`**: any positional arguments passed to `graph_func`.
- **`graph_filename`**: a file path to save the image. Default format is `png` but `jpg` and others are also supported.
- **`kwargs`**: any keyword arguments passed to `graph_func`.

Returns None.
