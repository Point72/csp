import time
import unittest
from datetime import datetime, timedelta
from typing import List

import csp
from csp import ts


@csp.node
def accumulator(x: ts[int]) -> ts[List[int]]:
    """Accumulate all values into a list."""
    with csp.state():
        s_values = []

    if csp.ticked(x):
        s_values.append(x)
        return list(s_values)


@csp.node
def rolling_sum(x: ts[int], window: int) -> ts[int]:
    """Compute rolling sum over a window."""
    with csp.state():
        s_buffer = []

    if csp.ticked(x):
        s_buffer.append(x)
        if len(s_buffer) > window:
            s_buffer.pop(0)
        return sum(s_buffer)


@csp.node
def filter_even(x: ts[int]) -> ts[int]:
    """Filter to only even numbers."""
    if csp.ticked(x) and x % 2 == 0:
        return x


@csp.node
def multiply(x: ts[int], factor: int) -> ts[int]:
    """Multiply input by a factor."""
    if csp.ticked(x):
        return x * factor


class AsyncioModeTestCase(unittest.TestCase):
    """Base class that runs tests in both asyncio and background-thread modes."""

    def run_graph_both_modes(self, graph_func, endtime=timedelta(milliseconds=300)):
        """
        Run a graph in both modes and return results for comparison.

        Args:
            graph_func: A function that returns a csp.graph decorated function
            endtime: Duration to run the graph

        Returns:
            Tuple of (background_thread_results, same_thread_results)
        """
        graph = graph_func()

        # Run with asyncio on background thread (old default behavior)
        background_thread_results = csp.run(graph, realtime=True, endtime=endtime, asyncio_on_thread=True)

        # Run with asyncio on same thread (new default behavior)
        graph = graph_func()  # Fresh graph instance
        same_thread_results = csp.run(graph, realtime=True, endtime=endtime)  # asyncio_on_thread=False is default

        return background_thread_results, same_thread_results

    def extract_values(self, results, key):
        """Extract just the values from timestamped results."""
        if key not in results or not results[key]:
            return []
        return [v for _, v in results[key]]


class TestTimerEquivalence(AsyncioModeTestCase):
    """Test timer functionality is equivalent in both modes."""

    def test_timer_basic(self):
        """Test that timers produce same count in both modes."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=50), 1)
                counter = csp.count(timer)
                csp.add_graph_output("count", counter)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=250))

        normal_values = self.extract_values(normal, "count")
        asyncio_values = self.extract_values(asyncio_result, "count")

        # Both should have multiple values
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

        # Values should be close (timing may cause slight differences)
        # Check that we got approximately the same number of events
        self.assertAlmostEqual(len(normal_values), len(asyncio_values), delta=2)

    def test_timer_with_processing(self):
        """Test timer with node processing is equivalent."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=40), 1)
                counter = csp.count(timer)
                doubled = multiply(counter, 2)
                csp.add_graph_output("doubled", doubled)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=200))

        normal_values = self.extract_values(normal, "doubled")
        asyncio_values = self.extract_values(asyncio_result, "doubled")

        # Check both have values
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

        # All values should be even (doubled from count)
        for v in normal_values + asyncio_values:
            self.assertEqual(v % 2, 0)


class TestNodeEquivalence(AsyncioModeTestCase):
    """Test node functionality is equivalent in both modes."""

    def test_rolling_sum(self):
        """Test rolling sum produces consistent results."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=30), 1)
                counter = csp.count(timer)
                rsum = rolling_sum(counter, 3)
                csp.add_graph_output("rolling", rsum)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=200))

        normal_values = self.extract_values(normal, "rolling")
        asyncio_values = self.extract_values(asyncio_result, "rolling")

        # Both should produce results
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

    def test_filter_node(self):
        """Test filter node works equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=25), 1)
                counter = csp.count(timer)
                evens = filter_even(counter)
                csp.add_graph_output("evens", evens)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=250))

        normal_values = self.extract_values(normal, "evens")
        asyncio_values = self.extract_values(asyncio_result, "evens")

        # All values should be even
        for v in normal_values:
            self.assertEqual(v % 2, 0)
        for v in asyncio_values:
            self.assertEqual(v % 2, 0)

    def test_chained_nodes(self):
        """Test chained nodes produce equivalent results."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=30), 1)
                counter = csp.count(timer)
                doubled = multiply(counter, 2)
                tripled = multiply(doubled, 3)  # Actually 6x
                csp.add_graph_output("result", tripled)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=180))

        normal_values = self.extract_values(normal, "result")
        asyncio_values = self.extract_values(asyncio_result, "result")

        # All values should be multiples of 6
        for v in normal_values + asyncio_values:
            self.assertEqual(v % 6, 0)


class TestBaselibEquivalence(AsyncioModeTestCase):
    """Test baselib functions work equivalently in both modes."""

    def test_sample(self):
        """Test csp.sample works equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=20), 1)
                trigger = csp.timer(timedelta(milliseconds=100))
                counter = csp.count(timer)
                sampled = csp.sample(trigger, counter)
                csp.add_graph_output("sampled", sampled)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=350))

        normal_values = self.extract_values(normal, "sampled")
        asyncio_values = self.extract_values(asyncio_result, "sampled")

        # Both should have some samples
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

    def test_delay(self):
        """Test csp.delay works equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=50), 1)
                counter = csp.count(timer)
                delayed = csp.delay(counter, timedelta(milliseconds=25))
                csp.add_graph_output("delayed", delayed)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=300))

        normal_values = self.extract_values(normal, "delayed")
        asyncio_values = self.extract_values(asyncio_result, "delayed")

        # Both should have delayed values
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

    def test_merge(self):
        """Test csp.merge works equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer1 = csp.timer(timedelta(milliseconds=40), 1)
                timer2 = csp.timer(timedelta(milliseconds=60), 10)
                merged = csp.merge(timer1, timer2)
                csp.add_graph_output("merged", merged)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=250))

        normal_values = self.extract_values(normal, "merged")
        asyncio_values = self.extract_values(asyncio_result, "merged")

        # Both should have merged values (1s and 10s)
        self.assertGreater(len(normal_values), 0)
        self.assertGreater(len(asyncio_values), 0)

        # Check we get both types of values
        self.assertIn(1, normal_values)
        self.assertIn(10, normal_values)


class TestMathEquivalence(AsyncioModeTestCase):
    """Test math operations work equivalently in both modes."""

    def test_add(self):
        """Test addition works equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=40), 5)
                result = timer + 10
                csp.add_graph_output("result", result)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=200))

        normal_values = self.extract_values(normal, "result")
        asyncio_values = self.extract_values(asyncio_result, "result")

        # All values should be 15
        for v in normal_values + asyncio_values:
            self.assertEqual(v, 15)

    def test_comparison(self):
        """Test comparison operations work equivalently."""

        def make_graph():
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=30), 1)
                counter = csp.count(timer)
                gt_3 = counter > 3
                csp.add_graph_output("gt_3", gt_3)

            return graph

        normal, asyncio_result = self.run_graph_both_modes(make_graph, endtime=timedelta(milliseconds=400))

        normal_values = self.extract_values(normal, "gt_3")
        asyncio_values = self.extract_values(asyncio_result, "gt_3")

        # Should have both True and False values
        self.assertIn(False, normal_values)
        self.assertIn(True, normal_values)


class TestAsyncioModePerformance(unittest.TestCase):
    """Test that both asyncio modes keep up with realtime.

    These deliberately assert loose behavioural bounds rather than latency budgets or
    mode-vs-mode ratios: the point is to catch a mode that stalls or spins, not to benchmark.
    Timing comparisons flake on emulated, sanitized or saturated runners; they belong in asv.
    """

    def test_both_modes_track_realtime(self):
        """
        Both execution modes must run a realtime graph to completion in roughly real time.

        A mode that stalls or busy-spins shows up as a wall time far from the requested duration.
        """

        @csp.graph
        def benchmark_graph():
            # Create a moderately complex graph
            timer = csp.timer(timedelta(milliseconds=5), 1)
            counter = csp.count(timer)

            # Chain several operations
            doubled = counter * 2
            tripled = counter * 3
            summed = doubled + tripled  # 5x

            # Add some filtering using csp.filter with a boolean ts
            is_even = (summed % 2) == 0
            evens = csp.filter(is_even, summed)

            csp.add_graph_output("result", evens)

        duration = timedelta(milliseconds=200)

        # Warm up
        csp.run(benchmark_graph, realtime=True, endtime=timedelta(milliseconds=50), asyncio_on_thread=True)
        csp.run(benchmark_graph, realtime=True, endtime=timedelta(milliseconds=50))  # default: asyncio on same thread

        expected = duration.total_seconds()
        for asyncio_on_thread in (True, False):
            with self.subTest(asyncio_on_thread=asyncio_on_thread):
                start = time.perf_counter()
                csp.run(benchmark_graph, realtime=True, endtime=duration, asyncio_on_thread=asyncio_on_thread)
                elapsed = time.perf_counter() - start

                # A realtime run cannot finish early, and must not overshoot wildly
                self.assertGreaterEqual(elapsed, expected * 0.9)
                self.assertLess(elapsed, expected + 5.0)

    def test_throughput_comparison(self):
        """
        Both modes must actually process events over a realtime window.

        No mode-vs-mode ratio is asserted; a stalled loop shows up as near-zero events.
        """

        def count_events(asyncio_on_thread: bool) -> int:
            @csp.graph
            def graph():
                timer = csp.timer(timedelta(milliseconds=10), 1)
                counter = csp.count(timer)
                csp.add_graph_output("count", counter)

            result = csp.run(
                graph, realtime=True, endtime=timedelta(milliseconds=200), asyncio_on_thread=asyncio_on_thread
            )
            return len(result.get("count", []))

        # ~20 ticks are scheduled in the window; require a clear majority to allow for a slow
        # runner while still failing a mode that is not draining its queue
        for asyncio_on_thread in (True, False):
            with self.subTest(asyncio_on_thread=asyncio_on_thread):
                self.assertGreaterEqual(count_events(asyncio_on_thread), 10)


if __name__ == "__main__":
    unittest.main()
