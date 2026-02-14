import base64
import gc
import pickle
import sys
import threading
from collections import defaultdict
from concurrent.futures import Future
from datetime import datetime
from functools import reduce
from io import BytesIO
from typing import Dict, List

import numpy as np

import csp
from csp.impl.genericpushadapter import GenericPushAdapter
from csp.impl.struct import Struct
from csp.impl.types.tstype import ts
from csp.impl.wiring.node import node

_CSS = """<style>
    body { background-color: #e8e8ed; font-family: proxima-nova,sans-serif; font-size: 1em; line-height: 1.5; font-weight: 400; }
               h3 { font-family: proxima-nova,sans-serif; font-size: 1.25em; line-height: 32px; color: #004a80; }
    .dataframe { font-family: proxima-nova,sans-serif; border-collapse: collapse; text-align: right; }
    table.dataframe, .dataframe th {
        padding: 4px 8px; font-weight: normal; font-size: 15px; color: #008AD0;
    background-color: #F2F3F4; text-align: right;}
    table.dataframe, .dataframe td {
        padding: 4px 8px; font-weight: normal; font-size: 13px; color: #2D2D2D;
    border-bottom: 1px solid black; text-align: right;}
    </style>"""

try:
    import tornado.ioloop
    import tornado.web

    HAS_TORNADO = True
    TornadoRequestHandler = tornado.web.RequestHandler
    TornadoWebApplication = tornado.web.Application
except ImportError:
    HAS_TORNADO = False

    class TornadoRequestHandler:
        pass

    class TornadoWebApplication:
        pass


def left_align(df):
    formatters = dict()
    # Left-align columns
    for col in df.select_dtypes("object"):
        len_max = df[col].str.len().max()
        formatters[col] = lambda _: f"{_:<{len_max}s}"
    return formatters


def write_image(handler, fig):
    tmpfile = BytesIO()
    fig.tight_layout()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    html = "<img src='data:image/png;base64,{}'>".format(encoded)
    # Write to HTML
    handler.write(html)
    handler.write("<br />")


class GraphInfo(Struct):
    node_count: int
    edge_count: int
    nodetype_counts: dict
    longest_path: list

    def print_info(self, sort_by: str = "count", max_nodes: int = 100):
        """
        Prints the GraphInfo object in a tabular form
        sort_by:     key to sort node data by. Valid keys are: "name", "count". Sorting by name is ascending (alphabetical) and sorting by count is descending.
        max_nodes:   the maximum number of nodes to print in the count table
        """

        print(self.format_info(sort_by, max_nodes))

    def format_info(self, sort_by, max_nodes):
        spacing = "-" * 90
        s = "\nGraphInfo results\n\n"
        s += f"Nodes: {self.node_count}\n"
        s += f"Edges: {self.edge_count}\n\n"
        s += spacing + "\n"
        s += "{0:30} {1}\n".format("Node", "Count")
        s += spacing + "\n"

        if sort_by == "name":
            node_data = sorted(list(self.nodetype_counts.items()), key=lambda x: x[0])
        elif sort_by == "count":
            node_data = sorted(list(self.nodetype_counts.items()), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f'Given sort_key {sort_by} is invalid: must be one of "name", "count"')

        for a_node, count in node_data[:max_nodes]:
            s += "{0:30} {1}\n".format(a_node, count)
        s += spacing + "\n"

        s += "\nLongest path\n"
        s += f"Size: {len(self.longest_path)}\n"
        s += f"Start: {self.longest_path[0]}\n"
        s += f"End: {self.longest_path[-1]}\n"
        s += f"Full Path: {self.longest_path}\n"

        return s

    def correct_from_profiler(self):
        # Correct for profiling nodes
        ignore_nodes = ["_profile", "_launch_application", "nullts"]
        for a_node in ignore_nodes:
            if a_node in self.nodetype_counts:
                self.node_count -= self.nodetype_counts[a_node]
                del self.nodetype_counts[a_node]
        self.edge_count -= 1  # connecting into the profiler
        return self

    def most_common_node(self):
        """
        Returns the most common nodetype in the graph as a tuple: (name, count)
        """
        return max(self.nodetype_counts.items(), key=lambda x: x[1])


def graph_info(g, *args, **kwargs):
    """
    Returns static information about the given csp.graph in the form of a GraphInfo object

    Input:
        graph: the csp graph to analyze

    Output: a GraphInfo object with the following attributes
        node_count: the number of nodes in the graph, including adapters
        edge_count: the number of node inputs in the graph
        nodetype_counts: a dictionary of node types (by name) to the number of those types in the graph
        longest_path: the longest path in the DAG, represented as a list of nodes
    """
    from csp.impl.wiring.context import Context
    from csp.impl.wiring.runtime import build_graph

    if not isinstance(g, Context):
        g = build_graph(g, *args, **kwargs)

    node_count = 0
    edge_count = 0
    nodetype_counts = dict()
    max_distance = dict()
    max_dist_child = dict()

    # longest path in a DAG: topological sort the graph (already done), then connect a dummy node to all roots and use dynamic programming
    # does not add any real work since we are traversing the graph anyways

    def dfs(nodedef):
        nonlocal node_count, edge_count, max_distance, max_dist_child

        if nodedef in max_distance:
            return max_distance[nodedef]

        max_distance[nodedef] = 0
        node_count += 1
        name = type(nodedef).__name__
        if name not in nodetype_counts:
            nodetype_counts[name] = 1
        else:
            nodetype_counts[name] += 1

        for v in nodedef.ts_inputs():
            edge_count += 1
            vdef = v[1].nodedef
            max_dist_v = dfs(vdef)
            if max_dist_v + 1 > max_distance[nodedef]:
                max_distance[nodedef] = max_dist_v + 1
                max_dist_child[nodedef] = vdef

        return max_distance[nodedef]

    for nodedef in g.roots:
        dfs(nodedef)

    if len(max_distance) == 0:  # empty graph
        return GraphInfo(node_count=0, edge_count=0, nodetype_counts=dict(), longest_path=list())

    # backtrack to get longest path
    curr = max(max_distance, key=max_distance.get)
    # lp_size = max_distance[curr]
    lp_path = list()
    while max_distance[curr] != 0:
        lp_path.append(type(curr).__name__)
        curr = max_dist_child[curr]
    lp_path.append(type(curr).__name__)

    return GraphInfo(
        node_count=node_count, edge_count=edge_count, nodetype_counts=nodetype_counts, longest_path=lp_path[::-1]
    )


class ProfilerInfo(Struct):
    cycle_count: int
    average_cycle_time: float
    max_cycle_time: float
    utilization: float
    node_stats: dict
    graph_info: GraphInfo
    build_time: float  # seconds

    def from_engine(self, p: Dict):
        """
        Convert from dictionary to class repr.
        """
        self.cycle_count = p["cycle_count"]
        self.average_cycle_time = p["average_cycle_time"]
        self.max_cycle_time = p["max_cycle_time"]
        self.utilization = p["utilization"]
        self.node_stats = p["node_stats"]
        if "_profile" in self.node_stats:
            del self.node_stats["_profile"]  # don't display profiling node times

    def print_stats(self, sort_by: str = "total_time", max_nodes: int = 100):
        """
        Prints profiling statistics for the csp graph
        sort_by:    key to sort node data by. Valid keys are: "name", "executions", "total_time", "max_time".
            --- All keys are presented in descending order except for name, which is ascending
        max_nodes:  the maximum number of nodes to display in the node data table
        """
        print(self.format_stats(sort_by, max_nodes))

    def format_stats(self, sort_by, max_nodes):
        sort_keys = ["name", "executions", "total_time", "max_time"]
        if sort_by not in sort_keys:
            raise ValueError(f"The given key {sort_by} is not valid: please choose one of {sort_keys}")

        # Print static info first
        s = ""
        if hasattr(self, "graph_info"):
            s = self.graph_info.format_info("count", 100)

        # Now print dynamic results
        spacing = "-" * 90
        s += f"\n{spacing}\n"
        s += "ProfilerInfo results:\n\n"
        s += f"Build time: {round(self.build_time, 4)} s \n\n"
        df = self._cycle_data_as_df()
        s += df.to_string(formatters=left_align(df), index=False, header=False)

        s += "\nNode execution data\n"
        s += f"\n{spacing}\n"
        df = self._node_data_as_df(sort_by, max_nodes)
        s += df.to_string(formatters=left_align(df), justify="left", index=False)

        return s

    def _cycle_data_as_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "names": [
                    "Number of cycles executed",
                    "Average cycle execution time (s)",
                    "Maximum cycle execution time (s)",
                    "Average graph utilization",
                ],
                "data": [
                    self.cycle_count,
                    round(self.average_cycle_time, 8),
                    round(self.max_cycle_time, 8),
                    round(self.utilization, 3),
                ],
            }
        )

    def _node_data_as_df(self, sort_by: str, max_nodes: int):
        import pandas as pd

        node_data = list(self.node_stats.items())
        if sort_by == "name":
            node_data.sort(key=lambda x: x[0])
        else:
            node_data.sort(key=lambda x: x[1][sort_by], reverse=True)

        total_time = sum([data["total_time"] for _, data in node_data])
        return pd.DataFrame(
            {
                "Node Type": [node for node, _ in node_data[:max_nodes]],
                "Executions": [data["executions"] for _, data in node_data[:max_nodes]],
                "Total Time (s)": [data["total_time"] for _, data in node_data[:max_nodes]],
                "Average Time (s)": [data["total_time"] / data["executions"] for _, data in node_data[:max_nodes]],
                "Max. Time (s)": [data["max_time"] for _, data in node_data[:max_nodes]],
                "% Total Time": [round(data["total_time"] / total_time * 100, 2) for _, data in node_data[:max_nodes]],
            }
        )

    def dump_stats(self, filename: str):
        """
        Writes the current profiler results to a file
        filename:   the file to write to
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_stats(self, filename: str):
        """
        Loads the profiler stats from a file and returns the ProfilerInfo object
        filename:   the file to load from
        """
        with open(filename, "rb") as f:
            p = pickle.load(f)
        return p

    def max_time_node(self):
        """
        Returns a tuple with the node that took the most amount of time throughout the run as (name, node_stats)
        """
        return max(self.node_stats.items(), key=lambda x: x[1]["total_time"])

    def max_exec_node(self):
        """
        Returns a tuple with the node that had the largest number of executions throughout the run as (name, node_stats)
        """
        return max(self.node_stats.items(), key=lambda x: x[1]["executions"])

    def memory_snapshot(self, max_size: int = 20):
        """
        Records the current memory usage per object type in existence
        """
        obj_info = defaultdict(lambda: [0, 0])  # count, size (B)

        gc.collect()
        for o in gc.get_objects():
            key = str(type(o))
            # This actually fkin leaks if you call getsizeof on unittest mock objects!!
            if key.startswith("unittest"):
                continue
            obj_info[key][0] += 1
            obj_info[key][1] += sys.getsizeof(o)

        total_size = reduce(lambda a, b: a + b[1], obj_info.values(), 0)
        obj_by_size = sorted(obj_info.items(), key=lambda x: x[1][1], reverse=True)
        return total_size, obj_by_size[:max_size]

    def _memory_data_as_df(self, obj_by_size: List):
        import pandas as pd

        names, data = zip(*obj_by_size)
        counts, size = zip(*data)
        return pd.DataFrame({"Object Type": names, "Object Count": counts, "Total Size (B)": size})

    def __sub__(self, other):
        if not isinstance(other, ProfilerInfo):
            raise TypeError("ProfilerInfo type is required")
        # Static values remain
        new_profile = ProfilerInfo(
            average_cycle_time=other.average_cycle_time,
            max_cycle_time=other.max_cycle_time,
            build_time=other.build_time,
        )
        new_profile.cycle_count = self.cycle_count - other.cycle_count
        new_profile.utilization = self.utilization - other.utilization
        new_profile.node_stats = {}
        for node_name, cur_node in self.node_stats.items():
            other_node = other.node_stats[node_name]
            execs = cur_node["executions"] - other_node["executions"]
            if execs == 0:
                continue
            new_node = new_profile.node_stats[node_name] = cur_node.copy()
            new_node["executions"] -= other_node["executions"]
            new_node["total_time"] -= other_node["total_time"]

        return new_profile


class ProfilerUIHandler(TornadoRequestHandler):
    def initialize(self, adapter: GenericPushAdapter, display_graphs: bool):
        self.adapter = adapter
        self.display_graphs = display_graphs
        if display_graphs:
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                raise Exception("You must have matplotlib installed to display profiling data graphs.")

    def get(self):
        try:
            # Get profiling data
            request = Future()
            self.adapter.push_tick(request)
            prof_info = request.result(timeout=10)

            if self.get_argument("format", "") == "json":
                tld = {"profiling_data": prof_info.to_dict(), "build_time": prof_info.build_time}
                if self.get_argument("snap_memory", "").lower() in ("1", "true"):
                    tld["memory_data"] = dict(prof_info.memory_snapshot()[1])
                self.write(tld)

            else:
                self.write(_CSS)
                # Display output in HTML table format
                sep = "-" * 90 + "<br /><br />"
                self.write(f"Current time: {datetime.now()}<br />")
                self.write(sep)

                # Write cycle level data
                self.write(f"Graph build time: {prof_info.build_time} s<br /><br />")
                self.write("Cycle Executions/Times<br /><br />")
                df = prof_info._cycle_data_as_df()
                self.write(df.to_html(formatters=left_align(df), index=False, header=False))
                self.write(sep)

                # Write node level data
                self.write("Node Executions/Time<br /><br />")
                df = prof_info._node_data_as_df("total_time", 100)
                self.write(df.to_html(formatters=left_align(df), justify="left", index=False))
                self.write(sep)

                if self.display_graphs:
                    from matplotlib.figure import Figure

                    # Write node bar chart
                    fig = Figure(facecolor="#e8e8ed")
                    ax = fig.subplots()
                    x_pos = np.arange(len(df["Node Type"]))
                    ax.set_xticks(x_pos, labels=df["Node Type"], rotation=90)
                    ax.bar(x_pos, df["Total Time (s)"], align="center")
                    ax.invert_xaxis()
                    ax.set_ylabel("Time (s)")
                    ax.set_title("Total Execution Time by Node Type")
                    write_image(self, fig)

                    # Write node pie chart
                    fig = Figure(facecolor="#e8e8ed")
                    ax = fig.subplots()
                    ax.pie(df["Total Time (s)"], labels=df["Node Type"], autopct="%1.1f%%")
                    ax.axis("equal")
                    ax.set_title("Total Execution Time by Node Type")
                    write_image(self, fig)
                    self.write(sep)

                # Write memory data
                self.write("Current Memory Usage<br /><br />")
                total, by_obj = prof_info.memory_snapshot()
                self.write(f"Total Usage: {total} bytes<br />")
                df = prof_info._memory_data_as_df(by_obj)
                self.write(df.to_html(formatters=left_align(df), justify="left", index=False))
                self.write(sep)

        except Exception as e:
            print(e)
            self.write("No profiler info available...")


class Profiler:
    TLS = threading.local()

    def __init__(self, http_port: int = None, cycle_file: str = "", node_file: str = "", display_graphs: bool = False):
        """
        If a port number is provided, then the Profiler Web UI will be available.
        """
        self.prof_info = ProfilerInfo()
        self.graph_info = GraphInfo()
        self.initialized = False
        self.http_port = http_port
        self.cycle_file = cycle_file
        self.node_file = node_file
        self.display_graphs = display_graphs

    def __enter__(self):
        self.TLS.instance = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del self.TLS.instance

    def init_profiler(self):
        if self.http_port is not None:
            if not HAS_TORNADO:
                raise Exception("You must have tornado installed to use the HTTP profiling extension.")

            adapter = GenericPushAdapter(Future)
            application = tornado.web.Application(
                [(r"/", ProfilerUIHandler, dict(adapter=adapter, display_graphs=self.display_graphs))],
                websocket_ping_interval=15,
            )
            _profile(self.prof_info, adapter.out())
            _launch_application(self.http_port, application)
        else:
            _profile(self.prof_info)

        if not self.initialized:  # for dynamic graphs
            self.build_starttime = datetime.now()

    @classmethod
    def instance(cls):
        return getattr(cls.TLS, "instance", None)

    def end_build(self):
        if not self.initialized:
            self.prof_info.build_time = (datetime.now() - self.build_starttime).total_seconds()
            self.initialized = True

    def results(self):
        self.prof_info.graph_info = self.graph_info
        return self.prof_info


@node
def nullts() -> ts[Future]:
    if False:
        return None


@node
def _profile(prof_info: ProfilerInfo, trigger: ts[Future] = nullts()):
    with csp.stop():
        prof_info.from_engine(csp.engine_stats())  # set the Profiler object in-memory

    """
    Returns runtime information about the given csp.graph in the form of a ProfilerInfo object

    Input:
        prof_info: mutable output container to store the profiling info
        trigger: an optional argument to trigger profiling results midway through a graph run

    Output: a ProfilerInfo object with the following attributes
        Graph properties
        cycle_count: number of cycles where any node in the graph ticked
        average_cycle_time: the average time for each cycle
        max_cycle_time: the maximum time for a single engine cycle
        utilization: the total number of times any node ticked throughout execution divided by the (number of nodes in the graph * the number of active engine cycles)
            -- represents the average fraction of the graph that is being executed each cycle

        Node properties
        node_stats: a dictionary where node_data[name] contains the profiling data for nodetype "name". This data is stored as a dictionary with keys:
            executions: the total number of times the nodetype was executed throughout the run
            max_time: the maximum amount of time it took for that nodetype to execute
            total_time: the total amount of time it took for that nodetype to execute throughout the run
    """

    if csp.ticked(trigger):
        prof_info.from_engine(csp.engine_stats())
        trigger.set_result(prof_info)


@node
def _launch_application(http_port: int, application: TornadoWebApplication):
    with csp.state():
        s_app = None
        s_ioloop = None
        s_iothread = None

    with csp.start():
        s_app = application
        s_app.listen(http_port)
        s_ioloop = tornado.ioloop.IOLoop.current()
        s_iothread = threading.Thread(target=s_ioloop.start)
        s_iothread.start()

    with csp.stop():
        if s_ioloop:
            s_ioloop.add_callback(s_ioloop.stop)
            if s_iothread:
                s_iothread.join()
