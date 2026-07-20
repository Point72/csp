import unittest
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.impl.wiring.csp_numba import numba_node


class TestNumbaNodeBenchmarks(unittest.TestCase):
    def test_benchmark_basket_of_structs(self):
        """Benchmark: numba_node vs csp.node for iterating over basket of struct signals.
        Run with: python -m pytest csp/tests/benchmark_numba_node.py::TestNumbaNodeBenchmarks::test_benchmark_basket_of_structs -xvs
        """
        from csp import profiler

        NUM_TICKS = 10000
        BASKET_SIZE = 20
        TICK_PERCENT = 0.5

        class Point(csp.Struct):
            x: float
            y: float

        @numba_node
        def sum_x_numba(points: list[ts[Point]]) -> ts[float]:
            total = 0.0
            for i in range(len(points)):
                if points[i].ticked():
                    total = total + points[i].x
            return total

        @csp.node
        def sum_x_python(points: [ts[Point]]) -> ts[float]:
            total = 0.0
            for i in range(len(points)):
                if csp.ticked(points[i]):
                    total += points[i].x
            return total

        num_ticking = max(1, int(BASKET_SIZE * TICK_PERCENT))
        curves_data = []
        for sig_idx in range(BASKET_SIZE):
            group = sig_idx // num_ticking
            ticks = [
                (
                    timedelta(milliseconds=t * (BASKET_SIZE // num_ticking) + group),
                    Point(x=float(t + sig_idx), y=float(t * 2)),
                )
                for t in range(NUM_TICKS)
            ]
            curves_data.append(ticks)

        @csp.graph
        def g_numba():
            signals = [csp.curve(Point, data) for data in curves_data]
            result = sum_x_numba(signals)
            csp.add_graph_output("result", result)

        with profiler.Profiler() as p:
            results_numba = csp.run(
                g_numba,
                starttime=datetime(2024, 1, 1),
                endtime=timedelta(seconds=NUM_TICKS),
            )
        prof_numba = p.results()
        numba_total_time = prof_numba.node_stats["PyNumbaNode"]["total_time"]
        numba_executions = prof_numba.node_stats["PyNumbaNode"]["executions"]

        @csp.graph
        def g_python():
            signals = [csp.curve(Point, data) for data in curves_data]
            result = sum_x_python(signals)
            csp.add_graph_output("result", result)

        with profiler.Profiler() as p:
            results_python = csp.run(
                g_python,
                starttime=datetime(2024, 1, 1),
                endtime=timedelta(seconds=NUM_TICKS),
            )
        prof_python = p.results()
        python_total_time = prof_python.node_stats["sum_x_python"]["total_time"]
        python_executions = prof_python.node_stats["sum_x_python"]["executions"]

        numba_results = [v for _, v in results_numba["result"]]
        python_results = [v for _, v in results_python["result"]]
        self.assertEqual(len(numba_results), len(python_results))
        for n, p in zip(numba_results, python_results):
            self.assertAlmostEqual(n, p, places=5)

        numba_latency_us = (numba_total_time / numba_executions) * 1_000_000
        python_latency_us = (python_total_time / python_executions) * 1_000_000
        speedup = python_total_time / numba_total_time if numba_total_time > 0 else float("inf")

        print(f"\n{'=' * 70}")
        print(
            f"BENCHMARK: Basket of structs ({NUM_TICKS} ticks, {BASKET_SIZE} signals, "
            f"{TICK_PERCENT * 100:.0f}% tick together)"
        )
        print(f"{'=' * 70}")
        print(f"  numba_node:  {numba_total_time:.4f}s total, {numba_latency_us:.2f}µs/exec ({numba_executions} execs)")
        print(
            f"  csp.node:    {python_total_time:.4f}s total, {python_latency_us:.2f}µs/exec ({python_executions} execs)"
        )
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    unittest.main()
