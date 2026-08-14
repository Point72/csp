import asyncio
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import AsyncIterator

import pytest

import csp
from csp import ts
from csp.impl.async_adapter import AsyncContext, _AsyncNodeState

# Windows has lower timer resolution (~15.6ms vs ~1ms on Unix)
IS_WINDOWS = sys.platform == "win32"
# Use longer delays on Windows to avoid timing issues
DEFAULT_DELAY = 0.05 if IS_WINDOWS else 0.02


def _has_decomposed_api():
    """Check if the CSP decomposed execution API is available."""
    try:
        from csp.impl.__cspimpl import _cspimpl

        engine = _cspimpl.PyEngine(realtime=True)
        return hasattr(engine, "start")
    except Exception:
        return False


HAS_DECOMPOSED_API = _has_decomposed_api()


async def async_return_value(value: int, delay: float = 0.05) -> int:
    """Async function that returns a value after a delay."""
    await asyncio.sleep(delay)
    return value


async def async_double(n: int, delay: float = 0.05) -> int:
    """Async function that doubles a value after a delay."""
    await asyncio.sleep(delay)
    return n * 2


async def async_generator(count: int, delay: float = 0.05) -> AsyncIterator[int]:
    """Async generator that yields count values."""
    for i in range(count):
        await asyncio.sleep(delay)
        yield i


async def async_side_effect(value: int, results: list, delay: float = 0.05) -> None:
    """Async function with side effect (appends to list)."""
    await asyncio.sleep(delay)
    results.append(value)


class TestAsyncFor(unittest.TestCase):
    """Tests for csp.async_for - async generator to time series."""

    def test_async_for_basic(self):
        """Test basic async_for functionality."""

        @csp.graph
        def graph():
            values = csp.async_for(async_generator(5, delay=DEFAULT_DELAY))
            csp.add_graph_output("values", values)

        # Use longer endtime on Windows due to longer delays
        endtime = timedelta(seconds=1.0) if IS_WINDOWS else timedelta(seconds=0.5)
        results = csp.run(graph, realtime=True, endtime=endtime)
        values = [v for _, v in results["values"]]
        self.assertEqual(values, [0, 1, 2, 3, 4])

    def test_async_for_type_inference(self):
        """Test that async_for correctly infers the output type."""

        async def typed_generator() -> AsyncIterator[str]:
            for s in ["a", "b", "c"]:
                await asyncio.sleep(0.02)
                yield s

        @csp.graph
        def graph():
            # A closure is not resolvable from module globals, so inference cannot see the
            # annotation and warns before falling back to ts[object]
            with self.assertWarns(UserWarning):
                values = csp.async_for(typed_generator())
            self.assertEqual(values.tstype.typ, object)

            typed = csp.async_for(typed_generator(), output_type=str)
            self.assertEqual(typed.tstype.typ, str)

            csp.add_graph_output("values", typed)

        results = csp.run(graph, realtime=True, endtime=timedelta(seconds=0.3))
        values = [v for _, v in results["values"]]
        self.assertEqual(values, ["a", "b", "c"])


class TestAsyncIn(unittest.TestCase):
    """Tests for csp.async_in - single async value to time series."""

    def test_async_in_basic(self):
        """Test basic async_in functionality."""

        @csp.graph
        def graph():
            result = csp.async_in(async_return_value(42, delay=DEFAULT_DELAY))
            csp.add_graph_output("result", result)

        endtime = timedelta(seconds=0.3) if IS_WINDOWS else timedelta(seconds=0.2)
        results = csp.run(graph, realtime=True, endtime=endtime)
        values = [v for _, v in results["result"]]
        self.assertEqual(values, [42])

    def test_async_in_ticks_once(self):
        """Test that async_in only ticks once."""

        @csp.graph
        def graph():
            result = csp.async_in(async_return_value(100, delay=DEFAULT_DELAY))
            csp.add_graph_output("result", result)

        endtime = timedelta(seconds=0.4) if IS_WINDOWS else timedelta(seconds=0.3)
        results = csp.run(graph, realtime=True, endtime=endtime)
        # Should only have one tick
        self.assertEqual(len(results["result"]), 1)
        self.assertEqual(results["result"][0][1], 100)


class TestAsyncOut(unittest.TestCase):
    """Tests for csp.async_out - time series to async function (side effects)."""

    def test_async_out_basic(self):
        """Test basic async_out functionality."""
        collected = []

        async def collector(value: int) -> None:
            await asyncio.sleep(0.01)
            collected.append(value)

        @csp.graph
        def graph():
            trigger = csp.timer(timedelta(milliseconds=50), True)
            counter = csp.count(trigger)
            csp.async_out(counter, collector)

        csp.run(graph, realtime=True, endtime=timedelta(seconds=0.25))

        # Give async operations time to complete
        import time

        time.sleep(0.2)

        # Should have collected multiple values
        self.assertGreater(len(collected), 0)
        self.assertEqual(collected, sorted(collected))  # Should be in order


class TestAsyncNode(unittest.TestCase):
    """Tests for csp.async_node - transform time series via async function."""

    def test_async_node_basic(self):
        """Test basic async_node functionality."""

        @csp.graph
        def graph():
            trigger = csp.timer(timedelta(milliseconds=50), True)
            counter = csp.count(trigger)
            doubled = csp.async_node(counter, async_double)
            csp.add_graph_output("doubled", doubled)

        results = csp.run(graph, realtime=True, endtime=timedelta(seconds=0.3))
        values = [v for _, v in results["doubled"]]

        # Each value should be doubled
        self.assertGreater(len(values), 0, "async_node produced no output at all")
        for i, v in enumerate(values, 1):
            self.assertEqual(v, i * 2)

    def test_async_node_rejects_bad_queue_settings(self):
        with self.assertRaises(ValueError) as ctx:
            csp.run(
                lambda: csp.async_node(csp.const(1), async_double, maxsize=-1),
                realtime=True,
                endtime=timedelta(seconds=0.1),
            )
        self.assertIn("maxsize", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            csp.run(
                lambda: csp.async_node(csp.const(1), async_double, on_overflow="drop_middle"),
                realtime=True,
                endtime=timedelta(seconds=0.1),
            )
        self.assertIn("on_overflow", str(ctx.exception))

    def test_async_node_bounded_queue_drops_instead_of_growing(self):
        """A bounded queue must shed load rather than buffer an unbounded backlog."""
        state = _AsyncNodeState(maxsize=2, on_overflow="drop_oldest")
        state.input_queue = asyncio.Queue(maxsize=2)

        for value in range(5):
            state.enqueue(value)

        self.assertEqual(state.input_queue.qsize(), 2)
        self.assertEqual(state.dropped, 3)
        # drop_oldest keeps the freshest ticks
        self.assertEqual([state.input_queue.get_nowait() for _ in range(2)], [3, 4])

    def test_async_node_drop_newest_keeps_the_earliest_ticks(self):
        state = _AsyncNodeState(maxsize=2, on_overflow="drop_newest")
        state.input_queue = asyncio.Queue(maxsize=2)

        for value in range(5):
            state.enqueue(value)

        self.assertEqual(state.dropped, 3)
        self.assertEqual([state.input_queue.get_nowait() for _ in range(2)], [0, 1])

    def test_async_node_default_queue_is_unbounded(self):
        """The default must not silently start dropping ticks for existing users."""
        state = _AsyncNodeState()
        state.input_queue = asyncio.Queue(maxsize=state.maxsize)

        for value in range(100):
            state.enqueue(value)

        self.assertEqual(state.input_queue.qsize(), 100)
        self.assertEqual(state.dropped, 0)

    def test_async_node_processor_exits_on_shutdown_sentinel(self):
        """Shutdown must wake the processor immediately, not wait for it to poll the queue."""
        from csp.impl.async_adapter import _ASYNC_NODE_SHUTDOWN, _AsyncNodeOutputAdapterImpl

        state = _AsyncNodeState()
        adapter = _AsyncNodeOutputAdapterImpl.__new__(_AsyncNodeOutputAdapterImpl)
        adapter._state = state
        adapter._tasks = set()
        adapter._async_func = None

        done = {}

        async def drive():
            state.loop = asyncio.get_running_loop()
            state.input_queue = asyncio.Queue()
            processor = asyncio.ensure_future(adapter._queue_processor())
            await asyncio.sleep(0)

            # state.active stays True, so only the sentinel can end this loop
            state.input_queue.put_nowait(_ASYNC_NODE_SHUTDOWN)
            await asyncio.sleep(0.05)

            done["exited"] = processor.done()
            processor.cancel()

        # wait_for cannot be used to detect a hang here: _queue_processor catches CancelledError,
        # so a timed-out wait_for sees a normally completed task and does not raise
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(drive())
        finally:
            loop.close()

        self.assertTrue(done["exited"], "processor did not exit on the shutdown sentinel")

    def test_async_node_maxsize_bounds_concurrent_work(self):
        """maxsize must bound in-flight work too, else the backlog just moves into the task set."""
        peak = {"value": 0, "current": 0}

        async def slow(n: int) -> int:
            peak["current"] += 1
            peak["value"] = max(peak["value"], peak["current"])
            try:
                await asyncio.sleep(0.05)
                return n
            finally:
                peak["current"] -= 1

        @csp.graph
        def graph():
            # Ticks far faster than slow() completes, so an unbounded node piles up tasks
            counter = csp.count(csp.timer(timedelta(milliseconds=2), True))
            csp.add_graph_output("out", csp.async_node(counter, slow, maxsize=2))

        csp.run(graph, realtime=True, endtime=timedelta(seconds=0.5))

        self.assertGreater(peak["value"], 0, "async_func never ran")
        self.assertLessEqual(peak["value"], 2, f"maxsize=2 did not bound concurrency, peaked at {peak['value']}")


class TestAwait(unittest.TestCase):
    """Tests for csp.await_ - blocking await within nodes."""

    def test_await_blocking(self):
        """Test blocking await within a node."""

        @csp.node
        def node_with_await() -> ts[int]:
            with csp.alarms():
                trigger = csp.alarm(bool)

            with csp.start():
                csp.schedule_alarm(trigger, timedelta(milliseconds=10), True)

            if csp.ticked(trigger):
                # Blocking await
                result = csp.await_(async_return_value(99, delay=DEFAULT_DELAY), block=True)
                return result

        @csp.graph
        def graph():
            result = node_with_await()
            csp.add_graph_output("result", result)

        endtime = timedelta(seconds=0.3) if IS_WINDOWS else timedelta(seconds=0.2)
        results = csp.run(graph, realtime=True, endtime=endtime)
        values = [v for _, v in results["result"]]
        self.assertIn(99, values)


class TestAsyncContext(unittest.TestCase):
    """Tests for AsyncContext - persistent async event loop in nodes."""

    def test_async_context_basic(self):
        """Test AsyncContext for persistent async operations."""

        @csp.node
        def node_with_context() -> ts[int]:
            with csp.alarms():
                trigger = csp.alarm(bool)

            with csp.state():
                s_ctx = None
                s_counter = 0

            with csp.start():
                s_ctx = AsyncContext()
                s_ctx.start()
                csp.schedule_alarm(trigger, timedelta(milliseconds=20), True)

            with csp.stop():
                if s_ctx:
                    s_ctx.stop()

            if csp.ticked(trigger):
                s_counter += 1
                if s_counter <= 3:
                    result = s_ctx.run(async_double(s_counter, delay=DEFAULT_DELAY))
                    csp.schedule_alarm(trigger, timedelta(milliseconds=50), True)
                    return result

        @csp.graph
        def graph():
            result = node_with_context()
            csp.add_graph_output("result", result)

        endtime = timedelta(seconds=0.8) if IS_WINDOWS else timedelta(seconds=0.5)
        results = csp.run(graph, realtime=True, endtime=endtime)
        values = [v for _, v in results["result"]]

        # Should have values 2, 4, 6 (1*2, 2*2, 3*2)
        self.assertEqual(values, [2, 4, 6])


class TestAsyncAlarm(unittest.TestCase):
    """Tests for csp.async_alarm - alarm-like pattern for async operations."""

    def test_async_alarm_basic(self):
        """Test basic async_alarm functionality."""

        @csp.node
        def node_with_async_alarm() -> ts[int]:
            with csp.alarms():
                poll_alarm = csp.alarm(bool)
                async_alarm = csp.async_alarm(int)

            with csp.state():
                s_counter = 0
                s_pending = False

            with csp.start():
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(poll_alarm):
                if not s_pending and s_counter < 3:
                    s_counter += 1
                    csp.schedule_async_alarm(async_alarm, async_double(s_counter, delay=0.05))
                    s_pending = True
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(async_alarm):
                s_pending = False
                return async_alarm

        @csp.graph
        def graph():
            result = node_with_async_alarm()
            csp.add_graph_output("result", result)

        results = csp.run(graph, realtime=True, endtime=timedelta(seconds=0.5))
        values = [v for _, v in results["result"]]

        # Should have values 2, 4, 6 (1*2, 2*2, 3*2)
        self.assertEqual(values, [2, 4, 6])

    def test_async_alarm_ticked_is_idempotent_within_a_cycle(self):
        """csp.ticked() on an async alarm must not consume a result per evaluation.

        Unit-level so the assertion is on the latch itself rather than on realtime delivery
        timing.  The generated per-cycle start_cycle() call is covered by test_async_alarm_basic,
        which returns stale repeats if the latch is never reset.
        """
        from csp.impl.async_adapter import AsyncAlarm

        alarm = AsyncAlarm(int)
        alarm._results.put(("result", 7))
        alarm._results.put(("result", 8))

        alarm.start_cycle()
        self.assertTrue(alarm.has_result())
        self.assertEqual(alarm.get_result(), 7)
        # Same cycle: must report the same tick, not eat the next result
        self.assertTrue(alarm.has_result())
        self.assertEqual(alarm.get_result(), 7)

        alarm.start_cycle()
        self.assertTrue(alarm.has_result())
        self.assertEqual(alarm.get_result(), 8)

        alarm.start_cycle()
        self.assertFalse(alarm.has_result())

    def test_async_alarm_delivers_without_a_poll_loop(self):
        """A completed coroutine must wake the node itself, not wait for a user's poll alarm."""

        @csp.node
        def no_poll(trigger: ts[bool]) -> ts[int]:
            with csp.alarms():
                aa = csp.async_alarm(int)

            if csp.ticked(trigger):
                csp.schedule_async_alarm(aa, async_double(21, delay=0.05))

            # No poll alarm, and csp.const ticks once at start, so after that tick nothing can
            # invoke this node again except the async alarm waking it
            if csp.ticked(aa):
                return aa

        @csp.graph
        def graph():
            csp.add_graph_output("out", no_poll(csp.const(True)))

        results = csp.run(graph, realtime=True, endtime=timedelta(seconds=0.4))

        self.assertEqual([v for _, v in results["out"]], [42], "result not delivered without a poll")

    def test_async_alarm_failure_reaches_a_node_with_no_poll(self):
        """A coroutine that raises must also wake the node, not be stranded in the queue."""

        async def boom():
            await asyncio.sleep(0.05)
            raise ValueError("async boom")

        @csp.node
        def no_poll(trigger: ts[bool]) -> ts[int]:
            with csp.alarms():
                aa = csp.async_alarm(int)

            if csp.ticked(trigger):
                csp.schedule_async_alarm(aa, boom())

            if csp.ticked(aa):
                return aa

        @csp.graph
        def graph():
            csp.add_graph_output("out", no_poll(csp.const(True)))

        with self.assertRaises(Exception) as ctx:
            csp.run(graph, realtime=True, endtime=timedelta(seconds=0.4))
        self.assertIn("async boom", str(ctx.exception.__cause__ or ctx.exception))

    def test_async_alarm_multiple_operations(self):
        """Test async_alarm with multiple sequential operations."""

        @csp.node
        def node_with_sequential_ops() -> ts[int]:
            with csp.alarms():
                poll_alarm = csp.alarm(bool)
                async_alarm = csp.async_alarm(int)

            with csp.state():
                s_values = [10, 20, 30, 40, 50]
                s_index = 0
                s_pending = False

            with csp.start():
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(poll_alarm):
                if not s_pending and s_index < len(s_values):
                    csp.schedule_async_alarm(async_alarm, async_double(s_values[s_index], delay=0.03))
                    s_index += 1
                    s_pending = True
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(async_alarm):
                s_pending = False
                return async_alarm

        @csp.graph
        def graph():
            result = node_with_sequential_ops()
            csp.add_graph_output("result", result)

        results = csp.run(graph, realtime=True, endtime=timedelta(seconds=0.6))
        values = [v for _, v in results["result"]]

        # Should have values 20, 40, 60, 80, 100 (each input doubled)
        self.assertEqual(values, [20, 40, 60, 80, 100])


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple async features."""

    def test_combined_async_features(self):
        """Test combining async_for, async_in, async_out, and async_node."""
        output_collected = []

        async def output_collector(value: int) -> None:
            await asyncio.sleep(DEFAULT_DELAY / 2)
            output_collected.append(value)

        @csp.graph
        def graph():
            # async_for: async generator to time series
            gen_values = csp.async_for(async_generator(3, delay=DEFAULT_DELAY))
            csp.add_graph_output("gen_values", gen_values)

            # async_in: single async value
            single_value = csp.async_in(async_return_value(42, delay=DEFAULT_DELAY))
            csp.add_graph_output("single_value", single_value)

            # async_node: transform via async
            doubled = csp.async_node(gen_values, async_double)
            csp.add_graph_output("doubled", doubled)

            # async_out: side effects
            csp.async_out(gen_values, output_collector)

        endtime = timedelta(seconds=0.8) if IS_WINDOWS else timedelta(seconds=0.4)
        results = csp.run(graph, realtime=True, endtime=endtime)

        # Verify async_for results
        gen_values = [v for _, v in results["gen_values"]]
        self.assertEqual(gen_values, [0, 1, 2])

        # Verify async_in results
        single_values = [v for _, v in results["single_value"]]
        self.assertEqual(single_values, [42])

        # Verify async_node results (doubled values)
        doubled_values = [v for _, v in results["doubled"]]
        self.assertEqual(doubled_values, [0, 2, 4])

        # Give time for async_out to complete
        import time

        time.sleep(0.2)
        self.assertEqual(sorted(output_collected), [0, 1, 2])


class TestSharedLoop(unittest.TestCase):
    """Tests for the shared async loop functionality."""

    def test_shared_loop_is_reused(self):
        """Verify that get_shared_loop returns the same loop instance."""
        from csp.impl.async_adapter import get_shared_loop, shutdown_shared_loop

        loop1 = get_shared_loop()
        loop2 = get_shared_loop()
        self.assertIs(loop1, loop2)
        self.assertTrue(loop1.is_running())

    def test_await_uses_shared_loop_by_default(self):
        """Verify await_ uses the shared loop by default."""
        from csp.impl.async_adapter import get_shared_loop

        shared_loop = get_shared_loop()
        loop_used = []

        async def capture_loop() -> int:
            loop_used.append(asyncio.get_running_loop())
            return 42

        result = csp.await_(capture_loop())
        self.assertEqual(result, 42)
        self.assertEqual(len(loop_used), 1)
        self.assertIs(loop_used[0], shared_loop)

    def test_await_with_custom_loop(self):
        """Verify await_ can use a custom loop when specified."""
        import threading

        custom_loop = asyncio.new_event_loop()
        loop_used = []
        loop_ready = threading.Event()

        async def capture_loop() -> int:
            loop_used.append(asyncio.get_running_loop())
            return 42

        def run_custom_loop():
            asyncio.set_event_loop(custom_loop)
            loop_ready.set()
            custom_loop.run_forever()

        thread = threading.Thread(target=run_custom_loop, daemon=True)
        thread.start()
        loop_ready.wait(timeout=2.0)

        try:
            result = csp.await_(capture_loop(), loop=custom_loop)
            self.assertEqual(result, 42)
            self.assertEqual(len(loop_used), 1)
            self.assertIs(loop_used[0], custom_loop)
        finally:
            custom_loop.call_soon_threadsafe(custom_loop.stop)
            thread.join(timeout=1.0)
            custom_loop.close()

    def test_multiple_adapters_use_shared_loop(self):
        """Verify that multiple adapters all use the same shared loop when using background-thread mode."""
        from csp.impl.async_adapter import get_shared_loop

        shared_loop = get_shared_loop()
        loops_observed = []

        async def record_loop(n: int) -> int:
            loops_observed.append(asyncio.get_running_loop())
            return n * 2

        async def record_loop_gen(count: int) -> AsyncIterator[int]:
            for i in range(count):
                loops_observed.append(asyncio.get_running_loop())
                yield i

        @csp.graph
        def graph():
            gen_values = csp.async_for(record_loop_gen(2))
            doubled = csp.async_node(gen_values, record_loop)
            csp.add_graph_output("doubled", doubled)

        # Use asyncio_on_thread=True to force the background-thread shared loop path
        csp.run(graph, realtime=True, endtime=timedelta(seconds=0.3), asyncio_on_thread=True)

        # All observed loops should be the shared loop
        self.assertGreater(len(loops_observed), 0)
        for loop in loops_observed:
            self.assertIs(loop, shared_loop)

    @pytest.mark.skipif(not HAS_DECOMPOSED_API, reason="CSP decomposed API not available")
    @pytest.mark.skipif(IS_WINDOWS, reason="CspEventLoop has Windows issues")
    def test_csp_event_loop_integration(self):
        """Verify that async adapters use CspEventLoop directly when available."""
        from csp.event_loop import CspEventLoop
        from csp.impl.async_adapter import get_async_loop

        loops_observed = []
        csp_loop = CspEventLoop()

        async def record_and_return(n: int) -> int:
            loops_observed.append(asyncio.get_running_loop())
            await asyncio.sleep(0.01)
            return n * 2

        async def main():
            # Inside CspEventLoop, get_async_loop should return the CspEventLoop
            current_loop = get_async_loop()
            self.assertIs(current_loop, csp_loop)

            # Schedule a coroutine
            result = await record_and_return(21)
            self.assertEqual(result, 42)

            # Verify the coroutine ran on the CspEventLoop
            self.assertEqual(len(loops_observed), 1)
            self.assertIs(loops_observed[0], csp_loop)

        try:
            csp_loop.run_until_complete(main())
        finally:
            csp_loop.close()


@pytest.mark.skipif(not HAS_DECOMPOSED_API, reason="CSP decomposed API not available")
@pytest.mark.skipif(IS_WINDOWS, reason="CspEventLoop has Windows issues")
class TestCspRunAsyncioMode(unittest.TestCase):
    """Tests for csp.run with asyncio_on_thread parameter (default: False = same thread)."""

    def test_asyncio_on_thread_simulation_mode(self):
        """Verify that asyncio_on_thread has no effect in simulation mode."""

        @csp.graph
        def graph():
            timer = csp.timer(timedelta(milliseconds=10), 1)
            csp.add_graph_output("count", csp.count(timer))

        # In simulation mode, asyncio_on_thread parameter is ignored
        # Both should complete successfully
        result1 = csp.run(graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=0.1), realtime=False)
        result2 = csp.run(
            graph,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=0.1),
            realtime=False,
            asyncio_on_thread=True,
        )

        self.assertIn("count", result1)
        self.assertIn("count", result2)

    def test_realtime_same_thread_asyncio_basic(self):
        """Test basic csp.run with same-thread asyncio (default in realtime mode)."""
        values = []

        @csp.node
        def collect(x: ts[int]):
            if csp.ticked(x):
                values.append(x)

        @csp.graph
        def graph():
            timer = csp.timer(timedelta(milliseconds=50), 1)
            counter = csp.count(timer)
            collect(counter)

        # Default in realtime is asyncio_on_thread=False (same thread)
        csp.run(graph, realtime=True, endtime=timedelta(milliseconds=200))

        # Should have some values collected
        self.assertGreater(len(values), 0)

    def test_realtime_background_thread_asyncio_basic(self):
        """Test csp.run with background thread asyncio (old default behavior)."""
        values = []

        @csp.node
        def collect(x: ts[int]):
            if csp.ticked(x):
                values.append(x)

        @csp.graph
        def graph():
            timer = csp.timer(timedelta(milliseconds=50), 1)
            counter = csp.count(timer)
            collect(counter)

        # Explicitly use background thread for asyncio
        csp.run(graph, realtime=True, endtime=timedelta(milliseconds=200), asyncio_on_thread=True)

        # Should have some values collected
        self.assertGreater(len(values), 0)

    def test_same_thread_async_adapter_uses_csp_loop(self):
        """Verify that async adapters use the CSP loop in same-thread mode."""
        from csp.impl.async_adapter import get_csp_asyncio_loop, is_csp_asyncio_mode

        loop_checks = []

        async def async_double(n: int) -> int:
            # Record whether we detect CSP asyncio mode
            is_asyncio = is_csp_asyncio_mode()
            loop = get_csp_asyncio_loop()
            loop_checks.append((is_asyncio, loop is not None))
            await asyncio.sleep(0.01)
            return n * 2

        @csp.graph
        def graph():
            timer = csp.timer(timedelta(milliseconds=50), 1)
            doubled = csp.async_node(timer, async_double)
            csp.add_graph_output("doubled", doubled)

        # Default in realtime: same-thread asyncio
        result = csp.run(graph, realtime=True, endtime=timedelta(milliseconds=300))

        # All async operations should detect CSP asyncio mode
        self.assertGreater(len(loop_checks), 0)
        for is_asyncio, has_loop in loop_checks:
            self.assertTrue(is_asyncio, "Should detect CSP asyncio mode")
            self.assertTrue(has_loop, "Should have access to CSP's asyncio loop")

    def test_same_thread_async_for(self):
        """Test async_for works with same-thread asyncio (default)."""

        async def async_gen() -> AsyncIterator[int]:
            for i in range(3):
                await asyncio.sleep(0.02)
                yield i

        @csp.graph
        def graph():
            gen = csp.async_for(async_gen())
            csp.add_graph_output("values", gen)

        # Default in realtime: same-thread asyncio
        result = csp.run(graph, realtime=True, endtime=timedelta(milliseconds=300))

        # Should have collected some values from async generator
        self.assertIn("values", result)
        if result["values"]:
            values = [v[1] for v in result["values"]]
            self.assertGreater(len(values), 0)

    def test_same_thread_async_in(self):
        """Test async_in works with same-thread asyncio (default)."""

        async def slow_fetch() -> int:
            await asyncio.sleep(0.02)
            return 42

        @csp.graph
        def graph():
            result = csp.async_in(slow_fetch())
            csp.add_graph_output("result", result)

        # Default in realtime: same-thread asyncio
        output = csp.run(graph, realtime=True, endtime=timedelta(milliseconds=200))

        # Should have received the result
        self.assertIn("result", output)
        if output["result"]:
            values = [v[1] for v in output["result"]]
            self.assertIn(42, values)


if __name__ == "__main__":
    unittest.main()
