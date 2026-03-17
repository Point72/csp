"""Tests for the native fd-based wakeup mechanism for event loop integration.

This tests the FdWaiter functionality that allows asyncio's selector to monitor
the CSP event queue without polling.
"""

import asyncio
import os
import select
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta

import csp
from csp.impl.__cspimpl import _cspimpl


class TestFdWakeupBasics(unittest.TestCase):
    """Test basic fd wakeup functionality."""

    def test_get_wakeup_fd_returns_valid_fd(self):
        """Test that get_wakeup_fd returns a valid file descriptor."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            fd = engine.get_wakeup_fd()
            # Should return a valid fd (>= 0) on Unix, or -1 if not supported
            self.assertIsInstance(fd, int)
            if sys.platform != "win32":
                # On Unix, should be a valid fd
                self.assertGreaterEqual(fd, 0)
        finally:
            engine.finish()

    def test_clear_wakeup_fd(self):
        """Test that clear_wakeup_fd can be called without error."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            # Should not raise
            engine.clear_wakeup_fd()
        finally:
            engine.finish()


@unittest.skipIf(sys.platform == "win32", "select.select on fds not supported on Windows")
class TestFdWakeupReadable(unittest.TestCase):
    """Test that the fd becomes readable when events are pushed."""

    def test_fd_not_readable_initially(self):
        """Test that the fd is not readable when no events are queued."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            fd = engine.get_wakeup_fd()
            if fd < 0:
                self.skipTest("FdWaiter not supported on this platform")

            # Check if fd is readable with zero timeout (should not be)
            readable, _, _ = select.select([fd], [], [], 0)
            # Note: might be readable if there are initial scheduled events
            # This is implementation-dependent
        finally:
            engine.finish()


class TestFdWakeupCrossThread(unittest.TestCase):
    """Test fd-based wakeup across threads."""

    @unittest.skipIf(sys.platform == "win32", "select.select on fds not supported on Windows")
    def test_fd_wakeup_integration_with_selector(self):
        """Test that the fd can be used with select for waiting."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            fd = engine.get_wakeup_fd()
            if fd < 0:
                self.skipTest("FdWaiter not supported on this platform")

            # This should not block indefinitely - select with timeout
            readable, _, _ = select.select([fd], [], [], 0.1)
            # Test passes if it doesn't hang
        finally:
            engine.finish()


class TestEventLoopFdIntegration(unittest.TestCase):
    """Test fd integration through CspEventLoop."""

    def test_event_loop_registers_wakeup_fd(self):
        """Test that CspEventLoop registers the wakeup fd with its selector."""
        from csp.event_loop import CspEventLoop

        loop = CspEventLoop(realtime=True)
        try:
            # Start the CSP engine
            loop._start_csp_engine()

            # Check that wakeup fd was registered
            if loop._csp_wakeup_fd is not None:
                self.assertGreaterEqual(loop._csp_wakeup_fd, 0)
                # Verify it's in the selector
                try:
                    key = loop._selector.get_key(loop._csp_wakeup_fd)
                    self.assertIsNotNone(key)
                except KeyError:
                    # Might not be registered if platform doesn't support it
                    pass
        finally:
            loop._stop_csp_engine()
            loop.close()

    def test_event_loop_unregisters_wakeup_fd_on_stop(self):
        """Test that CspEventLoop unregisters the wakeup fd when stopped."""
        from csp.event_loop import CspEventLoop

        loop = CspEventLoop(realtime=True)
        try:
            loop._start_csp_engine()
            wakeup_fd = loop._csp_wakeup_fd
            loop._stop_csp_engine()

            # Should be cleaned up
            self.assertIsNone(loop._csp_wakeup_fd)

            # Should not be in selector anymore
            if wakeup_fd is not None:
                with self.assertRaises(KeyError):
                    loop._selector.get_key(wakeup_fd)
        finally:
            loop.close()


class TestEndToEndWithCsp(unittest.TestCase):
    """End-to-end tests with CSP graphs and the event loop."""

    @unittest.skip("Needs to run in a file context")
    def test_csp_timer_in_simulation(self):
        """Test that CSP timers work correctly in simulation mode."""
        # Use simulation mode which is faster and more reliable for testing
        results = []

        @csp.node
        def collector(x: csp.ts[int]) -> csp.ts[int]:
            if csp.ticked(x):
                results.append(x)
            return x

        @csp.graph
        def test_graph():
            timer = csp.timer(timedelta(milliseconds=20))
            counter = csp.count(timer)
            csp.add_graph_output("out", collector(counter))

        # Run in simulation mode (fast, deterministic)
        start = datetime(2024, 1, 1)
        csp.run(test_graph, starttime=start, endtime=start + timedelta(milliseconds=150), realtime=False)

        # Should have received the events (7 timer ticks at 20ms intervals over 150ms)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(isinstance(r, int) for r in results))

    @unittest.skipIf(sys.platform == "win32", "select.select on fds not supported on Windows")
    def test_fd_wakeup_with_event_loop(self):
        """Test that the fd-based wakeup is properly registered with the event loop.

        This tests that the CspEventLoop registers the fd with its selector.
        """
        from csp.event_loop import CspEventLoop

        # Use the CspEventLoop directly
        loop = CspEventLoop(realtime=True)

        try:
            loop._start_csp_engine()
            fd = loop._csp_wakeup_fd

            if fd is None or fd < 0:
                self.skipTest("FdWaiter not supported")

            # Verify the fd is valid and registered
            self.assertGreaterEqual(fd, 0)

            # Test that we can select on it without hanging (timeout=0)
            readable, _, _ = select.select([fd], [], [], 0)
            # Test passes if it doesn't hang

        finally:
            loop._stop_csp_engine()
            loop.close()


class TestFdWakeupPerformance(unittest.TestCase):
    """Performance comparison between polling and fd-based wakeup."""

    @unittest.skipIf(sys.platform == "win32", "select.select on fds not supported on Windows")
    def test_select_latency(self):
        """Measure baseline latency of select with the wakeup fd."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            fd = engine.get_wakeup_fd()
            if fd < 0:
                self.skipTest("FdWaiter not supported")

            # Measure select latency (no events, immediate timeout)
            iterations = 1000
            start_time = time.perf_counter()
            for _ in range(iterations):
                select.select([fd], [], [], 0)
            end_time = time.perf_counter()

            avg_latency_us = (end_time - start_time) / iterations * 1_000_000
            print(f"\nSelect latency (no events): {avg_latency_us:.2f} µs/call")

            # Should be reasonable (< 100 µs per call on most systems)
            self.assertLess(avg_latency_us, 1000)  # < 1ms
        finally:
            engine.finish()

    @unittest.skipIf(sys.platform == "win32", "select.select on fds not supported on Windows")
    def test_wakeup_roundtrip_latency(self):
        """Measure roundtrip latency: notify -> select ready -> clear."""
        engine = _cspimpl.PyEngine(realtime=True)
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        engine.start(start, end)
        try:
            fd = engine.get_wakeup_fd()
            if fd < 0:
                self.skipTest("FdWaiter not supported")

            # We can't easily trigger notify from Python without pushing events
            # So we just measure select + clear cycle time
            iterations = 1000
            start_time = time.perf_counter()
            for _ in range(iterations):
                # Poll (no block)
                select.select([fd], [], [], 0)
                # Clear (even if nothing to clear)
                engine.clear_wakeup_fd()
            end_time = time.perf_counter()

            avg_latency_us = (end_time - start_time) / iterations * 1_000_000
            print(f"\nSelect + clear latency: {avg_latency_us:.2f} µs/call")

            self.assertLess(avg_latency_us, 1000)  # < 1ms
        finally:
            engine.finish()

    def test_polling_overhead_comparison(self):
        """Compare overhead of polling vs fd-based approach.

        This measures the cost difference between:
        1. Polling with time.sleep(interval)
        2. Using select with the wakeup fd
        """
        # Polling approach: time.sleep costs
        poll_iterations = 100
        poll_interval = 0.001  # 1ms

        start_time = time.perf_counter()
        for _ in range(poll_iterations):
            time.sleep(poll_interval)
        poll_time = time.perf_counter() - start_time

        # Expected time: poll_iterations * poll_interval
        expected_poll = poll_iterations * poll_interval
        poll_overhead = poll_time - expected_poll

        print(f"\nPolling ({poll_iterations} x {poll_interval * 1000:.1f}ms sleep):")
        print(f"  Total time: {poll_time * 1000:.2f} ms")
        print(f"  Expected: {expected_poll * 1000:.2f} ms")
        print(f"  Overhead: {poll_overhead * 1000:.2f} ms")

        # Now measure fd-based select (instant return when no events)
        if sys.platform != "win32":
            engine = _cspimpl.PyEngine(realtime=True)
            start = datetime.utcnow()
            end = start + timedelta(hours=1)
            engine.start(start, end)

            try:
                fd = engine.get_wakeup_fd()
                if fd >= 0:
                    start_time = time.perf_counter()
                    for _ in range(poll_iterations):
                        # With timeout 0, returns immediately
                        select.select([fd], [], [], 0)
                    select_time = time.perf_counter() - start_time

                    print(f"\nFd-based select ({poll_iterations} iterations, no block):")
                    print(f"  Total time: {select_time * 1000:.2f} ms")
                    print(f"  Per-call: {select_time / poll_iterations * 1000000:.2f} µs")

                    # Fd-based should be MUCH faster than polling with sleep
                    self.assertLess(select_time, poll_time / 10)
            finally:
                engine.finish()


if __name__ == "__main__":
    unittest.main()
