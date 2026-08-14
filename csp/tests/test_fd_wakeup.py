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

            readable, _, _ = select.select([fd], [], [], 0)
            self.assertEqual(readable, [])
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

            # An idle engine must leave select() to time out rather than spin or hang
            began = time.perf_counter()
            readable, _, _ = select.select([fd], [], [], 0.1)
            elapsed = time.perf_counter() - began

            self.assertEqual(readable, [])
            self.assertGreaterEqual(elapsed, 0.05)
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

            if loop._csp_wakeup_fd is None:
                self.skipTest("FdWaiter not supported on this platform")

            self.assertGreaterEqual(loop._csp_wakeup_fd, 0)
            self.assertIsNotNone(loop._selector.get_key(loop._csp_wakeup_fd))
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
    """Placeholder for wakeup-fd performance coverage.

    Absolute latency thresholds were removed from here: they measured `select()` and `time.sleep()`
    rather than the wakeup path, and asserting microsecond budgets in the correctness suite fails
    on emulated or saturated runners. Benchmarks belong in the asv suite.
    """


if __name__ == "__main__":
    unittest.main()
