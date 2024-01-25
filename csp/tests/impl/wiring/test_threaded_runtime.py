import time
import unittest
from datetime import datetime, timedelta

import csp
from csp import ts


@csp.node
def graph_stop(xs: ts["T"], output: dict) -> ts["T"]:
    with csp.stop():
        output["stop"] = csp.now()

    if csp.ticked(xs):
        return xs


@csp.node
def node_exception(xs: ts["T"]) -> ts["T"]:
    if csp.ticked(xs):
        raise ValueError("Test Node")
        return xs


@csp.graph
def graph_exception():
    raise ValueError("Test Graph")


class TestThreadedRuntime(unittest.TestCase):
    def test_run_on_thread(self):
        output = {}
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1)
        runner = csp.run_on_thread(
            graph_stop, csp.const(True), output, starttime=starttime, endtime=endtime, realtime=True
        )
        self.assertTrue(runner.is_alive())
        runner.stop_engine()
        runner.join()
        self.assertFalse(runner.is_alive())
        self.assertTrue(output.get("stop"))
        self.assertLess(output["stop"], endtime)

    def test_node_exception(self):
        runner = csp.run_on_thread(
            node_exception, csp.const(True), starttime=datetime.utcnow(), endtime=timedelta(seconds=1), realtime=True
        )
        self.assertRaises(RuntimeError, runner.join)
        self.assertFalse(runner.is_alive())
        self.assertEqual(runner.join(True), None)

    def test_graph_exception(self):
        runner = csp.run_on_thread(
            graph_exception, csp.const(True), starttime=datetime.utcnow(), endtime=timedelta(seconds=1), realtime=True
        )
        self.assertRaises(RuntimeError, runner.join)
        self.assertFalse(runner.is_alive())
        self.assertEqual(runner.join(True), None)

    def test_auto_shutdown(self):
        output = {}
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1)
        runner = csp.run_on_thread(
            graph_stop, csp.const(True), output, auto_shutdown=True, starttime=starttime, endtime=endtime, realtime=True
        )
        self.assertTrue(runner.is_alive())
        del runner
        self.assertTrue(output.get("stop"))
        self.assertLess(output["stop"], endtime)

    def test_auto_shutdown_exception(self):
        runner = csp.run_on_thread(
            node_exception,
            csp.const(True),
            auto_shutdown=True,
            starttime=datetime.utcnow(),
            endtime=timedelta(seconds=1),
            realtime=True,
        )
        time.sleep(1.0)  # Avoid race condition with graph raising exception
        del runner  # This should not throw an exception, even though one has been raised on the thread

    def test_csp_run_symmetric_api(self):
        # make sure args and kwargs passed the same

        @csp.graph
        def graph(arg: int):
            values = csp.const(arg)
            csp.print("values: ", values)

        # kwargs
        csp.run(graph, arg=1, starttime=datetime.utcnow(), endtime=timedelta(seconds=0.1), realtime=True)
        res = csp.run_on_thread(
            graph, arg=1, starttime=datetime.utcnow(), endtime=timedelta(seconds=0.1), realtime=True
        )
        res.join()

        # args
        csp.run(graph, 1, starttime=datetime.utcnow(), endtime=timedelta(seconds=0.1), realtime=True)
        res = csp.run_on_thread(graph, 1, starttime=datetime.utcnow(), endtime=timedelta(seconds=0.1), realtime=True)
        res.join()


if __name__ == "__main__":
    unittest.main()
