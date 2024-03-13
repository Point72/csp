## `csp.profiler()`

Users can simply run graphs under a `Profiler()` context to extract profiling information.
The code snippet below runs a graph in profile mode and extracts the profiling data by calling `results()`.

```python
from csp import profiler

with profiler.Profiler() as p:
    results = csp.run(graph, starttime=st, endtime=et) # run the graph normally

prof_data = p.results() # extract profiling data
```

Calling `results()` will return a ProfilerInfo object which contains the following attributes:

-  **`cycle_count`**: the total number of engine cycles executed during the run
-  **`average_cycle_time`**: the average time of a single engine cycle 
-  **`max_cycle_time`**: the maximum time of a single engine cycle 
-  **`utilization`**: the average "fraction of the graph" which runs each cycle. Defined as: (total number of node executions) / (number of nodes in the graph * cycle_count)`
-  **`node_stats`**: a dictionary where node_stats\["nodetype"\] holds the execution level data for each nodetype that was executed during the run. This data is stored as a dictionary with keys:
    - **`executions`**: the total number of times the nodetype was executed throughout the run
    - **`max_time`**: the maximum amount of time (in seconds) it took for that nodetype to execute
    - **`total_time`**: the total amount of time (in seconds) it took for that nodetype to execute throughout the run
- **`graph_info`**: a static GraphInfo object (see section 2 below) which is stored in the ProfilerInfo object
- **`build_time`**: the graph build time in seconds. This will be only computed if the graph is built under the profiler context i.e. the graph is not pre-built outside of the "with profiler" block

ProfilerInfo additionally comes with some useful utilities. These are:

- **`ProfilerInfo.print_stats(self, sort_by: str="total_time", max_nodes: int=100)`**
    - Prints profiling statistics in a table format for each node
        - **`sort_by`**: key to sort node data by. Valid keys are: "name", "executions", "total_time", "max_time". All keys are presented in descending order except for name, which is ascending.
        - **`max_nodes`**: the maximum number of nodes to display in the node data table. 
- **`ProfilerInfo.dump_stats(self, filename: str)`**
    - Writes the ProfilerInfo object to a file which can be shared and restored later.
        - **`filename`**: the filename to write to.
- **`ProfilerInfo.load_stats(self, filename: str)`**
    - Loads a ProfilerInfo object from a file. Returns the object.
        - **`filename`**: the filename to read from.
- **`ProfilerInfo.max_time_node(self)`**
    - Returns the node type which had the largest total execution time as a tuple: `(name, node_stat)` where node_stat is a dictionary with the same keys as `node_stats[elem]`
- **`ProfilerInfo.max_exec_node(self)`**
    - Returns the node type which had the most total executions as a tuple: `(name, node_stat)` where node_stat is a dictionary with the same keys as `node_stats[elem]`
