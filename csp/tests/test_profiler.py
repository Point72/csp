import os
import string
import sys
import tempfile
import time as Time
import unittest
from datetime import date, datetime, time, timedelta
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import pytz

import csp
import csp.stats as stats
from csp import profiler, ts
from csp.tests.test_dynamic import DynData, gen_basket, random_keys

from .test_showgraph import _cant_find_graphviz


@csp.graph
def stats_graph():
    # Larger graph, run for many cycles
    # Test: utilization, exec_counts, cycle_count
    # Also ensure: max and avg times are not equal for all nodes
    x = csp.count(csp.timer(timedelta(seconds=1)))
    y = csp.count(csp.timer(timedelta(seconds=2)))

    s = stats.sum(x, trigger=csp.timer(timedelta(seconds=5)))
    k = stats.kurt(y, trigger=csp.timer(timedelta(seconds=10)))

    csp.add_graph_output("s", s)
    csp.add_graph_output("k", k)


st = datetime(2020, 1, 1)


class TestProfiler(unittest.TestCase):
    def test_graph_info(self):
        @csp.graph
        def graph1():
            x0 = csp.timer(timedelta(seconds=1), 1)
            x1 = csp.timer(timedelta(seconds=1), 2)
            x2 = csp.timer(timedelta(seconds=1), 3)
            x3 = csp.timer(timedelta(seconds=1), True)

            x4 = csp.filter(x3, x0)
            x5 = csp.filter(x3, x1)
            x6 = csp.filter(x3, x2)

            x7 = csp.merge(x4, x5)
            x8 = csp.merge(x4, x6)

            csp.add_graph_output("x7", x7)
            csp.add_graph_output("x8", x8)

        graph_info = csp.profiler.graph_info(graph1)
        self.assertEqual(graph_info.node_count, 11)
        self.assertEqual(graph_info.edge_count, 12)
        self.assertEqual(len(graph_info.longest_path), 4)
        self.assertEqual(graph_info.longest_path[0], "csp.timer")
        self.assertEqual(graph_info.longest_path[-1], "add_graph_output")
        self.assertEqual(graph_info.most_common_node()[0], "csp.timer")
        self.assertEqual(graph_info.nodetype_counts["csp.timer"], 4)
        self.assertEqual(graph_info.nodetype_counts["filter"], 3)
        self.assertEqual(graph_info.nodetype_counts["merge"], 2)
        self.assertEqual(graph_info.nodetype_counts["add_graph_output"], 2)

        @csp.graph
        def graph_fb():
            # has some feedback connections
            x0 = csp.timer(timedelta(seconds=1), 5)

            reset_feedback = csp.feedback(bool)
            s = stats.sum(x0, 25, reset=reset_feedback.out())
            reset_signal = csp.gt(s, csp.const(100))  # reset sum whenever its greater than 100
            reset_feedback.bind(reset_signal)
            csp.add_graph_output("s", s)

        graph_info = csp.profiler.graph_info(graph_fb)
        self.assertEqual(graph_info.node_count, 15)
        self.assertEqual(graph_info.edge_count, 21)
        self.assertEqual(len(graph_info.longest_path), 7)
        self.assertEqual(graph_info.longest_path[0], "csp.timer")
        self.assertEqual(graph_info.longest_path[-1], "FeedbackOutputDef")
        self.assertEqual(graph_info.nodetype_counts["csp.timer"], 1)
        self.assertEqual(graph_info.nodetype_counts["FeedbackInputDef"], 1)
        self.assertEqual(graph_info.nodetype_counts["FeedbackOutputDef"], 1)

    def test_profile(self):
        @csp.node
        def sleep_for(t: ts[float]) -> ts[bool]:
            Time.sleep(t)
            return True

        # # # # # #
        # Test timing
        @csp.graph
        def graph1():
            # sleep for 1 second, 2 times
            x = csp.timer(timedelta(seconds=1), 1.0)
            sleep = sleep_for(x)
            csp.add_graph_output("sleep", sleep)

        with profiler.Profiler() as p:
            results = csp.run(graph1, starttime=st, endtime=st + timedelta(seconds=2))

        prof = p.results()

        epsilon = 0.0
        if sys.platform == "win32":
            epsilon = 0.05  # Clock resolution on windows is pretty bad
        self.assertGreater(prof.average_cycle_time + epsilon, 1.0)
        self.assertGreater(prof.max_cycle_time, 1.0)
        self.assertGreater(prof.node_stats["sleep_for"]["total_time"] + epsilon, 2.0)
        self.assertGreater(prof.node_stats["sleep_for"]["max_time"], 1.0)
        self.assertEqual(prof.node_stats["sleep_for"]["executions"], 2)
        self.assertEqual(prof.cycle_count, 2)
        self.assertEqual(prof.utilization, 1.0)  # profile node not included

        # tested to ensure profile=False afterwards

        with profiler.Profiler() as p:
            results = csp.run(stats_graph, starttime=st, endtime=st + timedelta(seconds=100))

        prof = p.results()

        # Cycles: 1 per seconds => 100
        # count, cast_int_to_float, time_window_updates: 1 per second + 1 per 2 seconds => 150
        # sum:  1 per 5 seconds + 1 at the first tick => 21
        # kurt: 1 per 10 seconds + 1 at the first tick => 11
        # total _compute execs: sum + kurt = 32
        # Util. = 482 / (9 * 100)

        self.assertEqual(prof.cycle_count, 100)
        self.assertEqual(prof.node_stats["count"]["executions"], 150)
        self.assertEqual(prof.node_stats["cast_int_to_float"]["executions"], 150)
        self.assertEqual(prof.node_stats["_time_window_updates"]["executions"], 150)
        self.assertEqual(prof.node_stats["_compute"]["executions"], 32)
        self.assertEqual(prof.utilization, 4.82 / 9)
        self.assertEqual(prof.graph_info, profiler.graph_info(stats_graph))

        # From test_dynamic.py
        @csp.graph
        def dyn(key: str, val: List[str], key_ts: ts[DynData], scalar: str):
            csp.add_graph_output(f"{key}_key", csp.const(key))
            csp.add_graph_output(f"{key}_val", csp.const(val))
            csp.add_graph_output(f"{key}_ts", key_ts)
            csp.add_graph_output(f"{key}_scalar", csp.const(scalar))
            key_ts = csp.merge(key_ts, csp.sample(csp.const(1), key_ts))
            csp.add_graph_output(f"{key}_tsadj", key_ts.val * 2)

        def graph3():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), True)
            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn, csp.snapkey(), csp.snap(keys), csp.attach(), "hello world!")

        with profiler.Profiler() as p:
            results = csp.run(graph3, starttime=st, endtime=st + timedelta(seconds=100))
            if not _cant_find_graphviz():
                with tempfile.NamedTemporaryFile(prefix="foo", suffix=".png", mode="w") as temp_file:
                    temp_file.close()
                    csp.show_graph(graph3, graph_filename=temp_file.name)

        prof = p.results()
        self.assertEqual(prof.cycle_count, 100)
        self.assertEqual(prof.node_stats["dynamic<dyn>"]["executions"], 100)
        self.assertEqual(prof.node_stats["gen_basket"]["executions"], 100)
        self.assertTrue("sample" in prof.node_stats)
        self.assertTrue("merge" in prof.node_stats)
        self.assertEqual("dynamic<dyn>", prof.max_time_node()[0])  # dynamic graph must take longest time as a "node"

        # test print stats
        prof.format_stats(sort_by="total_time", max_nodes=100)

        # test dump and load
        with tempfile.NamedTemporaryFile(prefix="foo", suffix=".p", mode="w") as temp_file:
            temp_file.close()

            prof.dump_stats(temp_file.name)
            p2 = prof.load_stats(temp_file.name)
        self.assertEqual(prof, p2)

    def test_node_names(self):
        # There was an issue where the baselib math ops were showing up with their generic names rather than overridden name
        with profiler.Profiler() as p:
            x = csp.const(1) + 2
            csp.run(x, starttime=datetime(2022, 8, 11), endtime=timedelta(seconds=1))

        self.assertTrue("add" in p.results().graph_info.nodetype_counts)
        # self.assertTrue('add' in p.results().node_stats )

        with profiler.Profiler() as p:
            x = csp.const("1") + "2"
            csp.run(x, starttime=datetime(2022, 8, 11), endtime=timedelta(seconds=1))

        self.assertTrue("add" in p.results().graph_info.nodetype_counts)
        self.assertTrue("add" in p.results().node_stats)

    def test_file_output(self):
        cycle_fn = f"cycle_data_{os.getpid()}.csv"
        node_fn = f"node_data_{os.getpid()}.csv"
        with profiler.Profiler(cycle_file=cycle_fn, node_file=node_fn) as p:
            results = csp.run(stats_graph, starttime=st, endtime=st + timedelta(seconds=100))

        # Verify that the files are proper, then clear them
        prof_info = p.results()

        with open(cycle_fn, "r") as f:
            lines = f.readlines()
            self.assertEqual(prof_info.cycle_count, len(lines[1:]))  # need to subtract one for column names
            file_act = sum([float(x) for x in lines[1:]]) / len(lines[1:])

        with open(node_fn, "r") as f:
            self.assertEqual(
                reduce(lambda a, b: a + b["executions"], prof_info.node_stats.values(), 0), len(f.readlines()) - 1
            )

        # Assert average cycle time is correct
        np.testing.assert_almost_equal(prof_info.average_cycle_time, file_act, decimal=6)

        # Make sure both can be read as csv
        df_node = pd.read_csv(node_fn)
        df_cycle = pd.read_csv(cycle_fn)
        max_times = df_node.groupby("Node Type").max().reset_index()
        self.assertEqual(
            round(prof_info.node_stats["cast_int_to_float"]["max_time"], 4),
            round(float(max_times.loc[max_times["Node Type"] == "cast_int_to_float"]["Execution Time"].iloc[0]), 4),
        )

        # Cleanup files
        os.remove(cycle_fn)
        os.remove(node_fn)

        # Ensure invalid file paths throw an error (do not fail silently)
        with self.assertRaises(ValueError):
            with profiler.Profiler(cycle_file="not_a_path/a.csv", node_file="also_not_a_path/b.csv") as p:
                results = csp.run(stats_graph, starttime=st, endtime=st + timedelta(seconds=100))


if __name__ == "__main__":
    unittest.main()
