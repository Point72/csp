The `csp.profiler` library allows users to time cycle/node executions during a graph run. There are two available utilities.

One can use these metrics to identify bottlenecks/inefficiencies in their graphs.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Profiling a real-time csp.graph](#profiling-a-real-time-cspgraph)
- [Saving raw profiling data to a file](#saving-raw-profiling-data-to-a-file)
- [graph_info: build-time information](#graph_info-build-time-information)

## Profiling a real-time `csp.graph`

The `csp.profiler` library provides a GUI for profiling real-time CSP graphs.
One can access this GUI by adding a `http_port` argument to their profiler call.

```python
with profiler.Profiler(http_port=8888) as p:
    results = csp.run(graph, starttime=st, endtime=et) # run the graph normally
```

This will open up the GUI on `localhost:8888` (as http_port=8888) which will display real-time node timing, cycle timing and memory snapshots.
Profiling stats will be calculated whenever you refresh the page or call a GET request.
Additionally, you can add the `format=json`argument (`localhost:8888?format=json`) to your request to receive the ProfilerInfo as a `JSON` object rather than the `HTML` display.

Users can add the `display_graphs=True` flag to include bar/pie charts of node execution times in the web UI.
The matplotlib package is required to use the flag.

```python
with profiler.Profiler(http_port=8888, display_graphs=True) as p:
    ...
```

<img width="466" alt="new_profiler" src="https://github.com/Point72/csp/assets/3105306/6ef692d2-16c3-4adb-ad46-a72e1017aa79">

## Saving raw profiling data to a file

Users can save individual node execution times and individual cycle execution times to a `.csv` file if they desire.
This is useful if you want to apply your own analysis e.g. calculate percentiles.
To do this, simply add the flags `node_file=<filename.csv>` or `cycle_file=<filename.csv>`

```python
with profiler.Profiler(cycle_file="cycle_data.csv", node_file="node_data.csv") as p:
    ...
```

After the graph is run, the file `node_data.csv` contains:

```
Node Type,Execution Time
count,1.9814e-05
cast_int_to_float,1.2791e-05
_time_window_updates,4.759e-06
...
```

After the graph is run, the file `cycle_data.csv` contains:

```
Execution Time
9.4757e-05
4.5205e-05
2.2873e-05
...
```

## graph_info: build-time information

Users can also extract build-time information about the graph without running it by calling `profiler.graph_info`.

The code snippet below shows how to call `graph_info`.

```python
from csp import profiler

info = profiler.graph_info(graph)
```
