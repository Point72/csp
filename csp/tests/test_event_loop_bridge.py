"""
Tests for CSP Event Loop Bridge Integration

This module tests the integration between asyncio and running CSP graphs
using the AsyncioBridge and BidirectionalBridge classes.
"""

import asyncio
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.event_loop.bridge import AsyncioBridge, BidirectionalBridge
from csp.utils.datetime import utc_now

# Windows has lower timer resolution (~15.6ms vs ~1ms on Unix)
IS_WINDOWS = sys.platform == "win32"
TIMING_TOLERANCE = 0.05 if IS_WINDOWS else 0.01


class TestAsyncioBridgeBasic(unittest.TestCase):
    """Basic tests for AsyncioBridge."""

    def test_create_bridge(self):
        """Test creating a bridge."""
        bridge = AsyncioBridge(int, "test_bridge")
        self.assertIsNotNone(bridge.adapter)
        self.assertFalse(bridge.is_running)
        self.assertIsNone(bridge.loop)

    def test_start_stop(self):
        """Test starting and stopping the bridge."""
        bridge = AsyncioBridge(int, "test_bridge")

        bridge.start()
        self.assertTrue(bridge.is_running)
        self.assertIsNotNone(bridge.loop)

        bridge.stop()
        self.assertFalse(bridge.is_running)
        self.assertIsNone(bridge.loop)

    def test_start_with_time(self):
        """Test starting with a specific start time."""
        bridge = AsyncioBridge(int, "test_bridge")
        start_time = datetime(2026, 1, 1, 12, 0, 0)

        bridge.start(start_time)
        self.assertEqual(bridge._start_time, start_time)

        bridge.stop()

    def test_double_start_raises(self):
        """Test that starting twice raises an error."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        try:
            with self.assertRaises(RuntimeError):
                bridge.start()
        finally:
            bridge.stop()

    def test_push_before_start_fails(self):
        """Test that push returns False before graph starts."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        # Push before adapter is bound to graph
        result = bridge.push(42)
        self.assertFalse(result)

        bridge.stop()


class TestAsyncioBridgeCallbacks(unittest.TestCase):
    """Tests for callback scheduling methods."""

    def test_call_soon(self):
        """Test call_soon schedules a callback."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        results = []

        bridge.call_soon(lambda: results.append("called"))

        # Wait for callback to execute
        time.sleep(0.1)

        self.assertEqual(results, ["called"])
        bridge.stop()

    def test_call_soon_with_args(self):
        """Test call_soon with arguments."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        results = []

        bridge.call_soon(lambda x, y: results.append(x + y), 1, 2)

        time.sleep(0.1)

        self.assertEqual(results, [3])
        bridge.stop()

    def test_call_soon_before_start_raises(self):
        """Test that call_soon before start raises an error."""
        bridge = AsyncioBridge(int, "test_bridge")

        with self.assertRaises(RuntimeError):
            bridge.call_soon(lambda: None)

    def test_call_later(self):
        """Test call_later schedules a delayed callback."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        results = []
        start = time.time()

        bridge.call_later(0.1, lambda: results.append(time.time() - start))

        # Wait for callback
        time.sleep(0.2)

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0], 0.09)

        bridge.stop()

    def test_call_later_negative_raises(self):
        """Test that negative delay raises an error."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        try:
            with self.assertRaises(ValueError):
                bridge.call_later(-1.0, lambda: None)
        finally:
            bridge.stop()

    def test_call_at(self):
        """Test call_at schedules at specific time."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        results = []
        target = datetime.utcnow() + timedelta(milliseconds=100)

        bridge.call_at(target, lambda: results.append(datetime.utcnow()))

        time.sleep(0.2)

        self.assertEqual(len(results), 1)
        # Allow some tolerance
        self.assertLess(abs((results[0] - target).total_seconds()), 0.05)

        bridge.stop()

    def test_call_at_past_time(self):
        """Test call_at with past time schedules immediately."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        results = []
        past_time = datetime.utcnow() - timedelta(seconds=10)

        bridge.call_at(past_time, lambda: results.append(True))

        time.sleep(0.1)

        self.assertEqual(results, [True])
        bridge.stop()

    def test_call_at_offset(self):
        """Test call_at_offset from start time."""
        bridge = AsyncioBridge(int, "test_bridge")
        start_time = datetime.utcnow()
        bridge.start(start_time)

        results = []

        bridge.call_at_offset(timedelta(milliseconds=100), lambda: results.append(True))

        time.sleep(0.2)

        self.assertEqual(results, [True])
        bridge.stop()


class TestAsyncioBridgeCoroutines(unittest.TestCase):
    """Tests for running coroutines."""

    def test_run_coroutine(self):
        """Test running a coroutine."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        async def my_coro():
            await asyncio.sleep(0.05)
            return 42

        future = bridge.run_coroutine(my_coro())
        result = future.result(timeout=1.0)

        self.assertEqual(result, 42)
        bridge.stop()

    def test_run_coroutine_before_start_raises(self):
        """Test that running coroutine before start raises."""
        bridge = AsyncioBridge(int, "test_bridge")

        async def my_coro():
            return 42

        with self.assertRaises(RuntimeError):
            bridge.run_coroutine(my_coro())

    def test_run_coroutine_with_exception(self):
        """Test coroutine that raises exception."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        async def failing_coro():
            raise ValueError("test error")

        future = bridge.run_coroutine(failing_coro())

        with self.assertRaises(ValueError):
            future.result(timeout=1.0)

        bridge.stop()


class TestAsyncioBridgeTime(unittest.TestCase):
    """Tests for time-related methods."""

    def test_time(self):
        """Test time() returns current time."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        t1 = bridge.time()
        time.sleep(0.05)
        t2 = bridge.time()

        self.assertGreater(t2, t1)
        self.assertGreaterEqual(t2 - t1, 0.04)

        bridge.stop()

    def test_elapsed_since_start(self):
        """Test elapsed_since_start returns correct duration."""
        bridge = AsyncioBridge(int, "test_bridge")
        bridge.start()

        time.sleep(0.1)
        elapsed = bridge.elapsed_since_start()

        self.assertGreaterEqual(elapsed.total_seconds(), 0.09)

        bridge.stop()

    def test_elapsed_before_start(self):
        """Test elapsed_since_start before start returns zero."""
        bridge = AsyncioBridge(int, "test_bridge")
        elapsed = bridge.elapsed_since_start()
        self.assertEqual(elapsed, timedelta(0))


class TestAsyncioBridgeWithCSP(unittest.TestCase):
    """Tests for AsyncioBridge with actual CSP graphs."""

    def test_push_to_csp(self):
        """Test pushing data from asyncio to CSP."""
        bridge = AsyncioBridge(int, "data")
        collected = []

        @csp.node
        def collect(data: ts[int]) -> ts[int]:
            if csp.ticked(data):
                return data

        @csp.graph
        def g():
            data = bridge.adapter.out()
            result = collect(data)
            csp.add_graph_output("data", result)

        start_time = utc_now()
        bridge.start(start_time)
        time.sleep(0.05)

        runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=1))

        bridge.wait_for_adapter(timeout=1.0)

        # Push some data
        for i in range(5):
            bridge.push(i)
            time.sleep(0.05)

        results = runner.join()
        bridge.stop()

        data = results.get("data", [])
        values = [v for _, v in data]
        self.assertEqual(values, [0, 1, 2, 3, 4])

    def test_call_later_with_csp(self):
        """Test call_later scheduling with CSP graph."""
        bridge = AsyncioBridge(str, "messages")

        @csp.node
        def collect(data: ts[str]) -> ts[str]:
            if csp.ticked(data):
                return data

        @csp.graph
        def g():
            data = bridge.adapter.out()
            result = collect(data)
            csp.add_graph_output("messages", result)

        start_time = utc_now()
        bridge.start(start_time)
        time.sleep(0.05)

        runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=1))

        bridge.wait_for_adapter(timeout=1.0)

        # Schedule messages with call_later
        bridge.call_later(0.1, lambda: bridge.push("msg1"))
        bridge.call_later(0.2, lambda: bridge.push("msg2"))
        bridge.call_later(0.3, lambda: bridge.push("msg3"))

        results = runner.join()
        bridge.stop()

        data = results.get("messages", [])
        values = [v for _, v in data]
        self.assertEqual(values, ["msg1", "msg2", "msg3"])

    def test_async_coroutine_with_csp(self):
        """Test running async coroutine that pushes to CSP."""
        bridge = AsyncioBridge(dict, "async_data")

        @csp.node
        def collect(data: ts[dict]) -> ts[dict]:
            if csp.ticked(data):
                return data

        @csp.graph
        def g():
            data = bridge.adapter.out()
            result = collect(data)
            csp.add_graph_output("data", result)

        start_time = utc_now()
        bridge.start(start_time)
        time.sleep(0.05)

        runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=1))

        bridge.wait_for_adapter(timeout=1.0)

        # Run coroutine that pushes to CSP
        async def fetch_and_push():
            for i in range(3):
                await asyncio.sleep(0.05)
                bridge.push({"value": i})

        bridge.run_coroutine(fetch_and_push())

        results = runner.join()
        bridge.stop()

        data = results.get("data", [])
        values = [v["value"] for _, v in data]
        self.assertEqual(values, [0, 1, 2])

    def test_csp_timer_with_async(self):
        """Test CSP timer running alongside async callbacks."""
        bridge = AsyncioBridge(str, "async")

        @csp.node
        def combine(timer: ts[bool], async_data: ts[str]) -> ts[str]:
            if csp.ticked(timer):
                return "timer"
            if csp.ticked(async_data):
                return f"async:{async_data}"

        @csp.graph
        def g():
            timer = csp.timer(timedelta(milliseconds=100))
            async_data = bridge.adapter.out()
            result = combine(timer, async_data)
            csp.add_graph_output("events", result)

        start_time = utc_now()
        bridge.start(start_time)
        time.sleep(0.05)

        runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(milliseconds=500))

        bridge.wait_for_adapter(timeout=1.0)

        # Push async data between timer ticks
        bridge.call_later(0.05, lambda: bridge.push("a"))
        bridge.call_later(0.15, lambda: bridge.push("b"))
        bridge.call_later(0.25, lambda: bridge.push("c"))

        results = runner.join()
        bridge.stop()

        data = results.get("events", [])
        values = [v for _, v in data]

        # Should have mix of timer and async events
        timer_count = sum(1 for v in values if v == "timer")
        async_count = sum(1 for v in values if v.startswith("async:"))

        self.assertGreater(timer_count, 0)
        self.assertGreater(async_count, 0)


class TestBidirectionalBridge(unittest.TestCase):
    """Tests for BidirectionalBridge."""

    def test_create_bidirectional(self):
        """Test creating a bidirectional bridge."""
        bridge = BidirectionalBridge(str, "bidi")
        self.assertIsNotNone(bridge.adapter)

    def test_on_event(self):
        """Test registering event callbacks."""
        bridge = BidirectionalBridge(str, "bidi")
        bridge.start()

        received = []
        bridge.on_event(lambda x: received.append(x))

        # Emit from "CSP side"
        bridge.emit({"test": 123})

        time.sleep(0.1)

        self.assertEqual(received, [{"test": 123}])
        bridge.stop()

    def test_off_event(self):
        """Test unregistering event callbacks."""
        bridge = BidirectionalBridge(str, "bidi")
        bridge.start()

        received = []
        callback = lambda x: received.append(x)

        bridge.on_event(callback)
        result = bridge.off_event(callback)
        self.assertTrue(result)

        bridge.emit({"test": 123})
        time.sleep(0.1)

        self.assertEqual(received, [])
        bridge.stop()

    def test_off_event_not_found(self):
        """Test unregistering non-existent callback."""
        bridge = BidirectionalBridge(str, "bidi")

        result = bridge.off_event(lambda x: None)
        self.assertFalse(result)

    def test_bidirectional_with_csp(self):
        """Test bidirectional communication with CSP."""
        bridge = BidirectionalBridge(str, "messages")
        received_in_async = []

        @csp.node
        def process_and_respond(data: ts[str], bridge_ref: object) -> ts[str]:
            if csp.ticked(data):
                response = f"processed:{data}"
                bridge_ref.emit({"original": data, "response": response})
                return response

        @csp.graph
        def g():
            data = bridge.adapter.out()
            result = process_and_respond(data, bridge)
            csp.add_graph_output("results", result)

        bridge.on_event(lambda x: received_in_async.append(x))

        start_time = utc_now()
        bridge.start(start_time)
        time.sleep(0.05)

        runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=1))

        bridge.wait_for_adapter(timeout=1.0)

        # Push data to CSP
        bridge.push("hello")
        time.sleep(0.1)
        bridge.push("world")
        time.sleep(0.1)

        results = runner.join()
        bridge.stop()

        # Check CSP outputs
        data = results.get("results", [])
        values = [v for _, v in data]
        self.assertEqual(values, ["processed:hello", "processed:world"])

        # Check asyncio received events
        self.assertEqual(len(received_in_async), 2)
        self.assertEqual(received_in_async[0]["original"], "hello")
        self.assertEqual(received_in_async[1]["original"], "world")

    def test_multiple_event_callbacks(self):
        """Test multiple event callbacks receive events."""
        bridge = BidirectionalBridge(str, "bidi")
        bridge.start()

        received1 = []
        received2 = []

        bridge.on_event(lambda x: received1.append(x))
        bridge.on_event(lambda x: received2.append(x))

        bridge.emit("test")
        time.sleep(0.1)

        self.assertEqual(received1, ["test"])
        self.assertEqual(received2, ["test"])

        bridge.stop()


class TestDeferredHandle(unittest.TestCase):
    """Tests for _DeferredHandle."""

    def test_cancel(self):
        """Test cancelling a deferred handle."""
        bridge = AsyncioBridge(int, "test")
        bridge.start()

        results = []
        handle = bridge.call_later(0.5, lambda: results.append(True))

        # Cancel before it fires
        handle.cancel()
        self.assertTrue(handle.cancelled())

        time.sleep(0.6)

        # Should not have been called
        self.assertEqual(results, [])

        bridge.stop()


if __name__ == "__main__":
    unittest.main()
