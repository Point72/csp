"""
Tests for CSP Event Loop Integration

This module tests the integration between CSP's event loop and Python's asyncio.
"""

import asyncio
import concurrent.futures
import socket
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

import pytest

import csp
from csp.event_loop import CspEventLoop, CspEventLoopPolicy, new_event_loop, run


class TestCspEventLoopBasic(unittest.TestCase):
    """Basic tests for CspEventLoop."""

    def test_create_loop(self):
        """Test creating a new event loop."""
        loop = new_event_loop()
        self.assertIsInstance(loop, CspEventLoop)
        self.assertFalse(loop.is_running())
        self.assertFalse(loop.is_closed())
        loop.close()
        self.assertTrue(loop.is_closed())

    def test_run_until_complete_simple(self):
        """Test run_until_complete with a simple coroutine."""

        async def simple_coro():
            return 42

        loop = new_event_loop()
        try:
            result = loop.run_until_complete(simple_coro())
            self.assertEqual(result, 42)
        finally:
            loop.close()

    def test_run_until_complete_with_await(self):
        """Test run_until_complete with a coroutine that awaits."""

        async def coro_with_await():
            await asyncio.sleep(0.01)
            return "done"

        loop = new_event_loop()
        try:
            result = loop.run_until_complete(coro_with_await())
            self.assertEqual(result, "done")
        finally:
            loop.close()

    def test_run_function(self):
        """Test the run() convenience function."""

        async def main():
            await asyncio.sleep(0.01)
            return "result"

        result = run(main())
        self.assertEqual(result, "result")

    def test_stop(self):
        """Test stopping the loop."""
        loop = new_event_loop()

        async def stopper():
            await asyncio.sleep(0.01)
            loop.stop()

        try:
            loop.create_task(stopper())
            loop.run_forever()
            # If we get here, stop worked
            self.assertFalse(loop.is_running())
        finally:
            loop.close()


class TestCallbackScheduling(unittest.TestCase):
    """Tests for callback scheduling methods."""

    def setUp(self):
        self.loop = new_event_loop()
        self.calls = []

    def tearDown(self):
        self.loop.close()

    def test_call_soon(self):
        """Test call_soon schedules a callback."""

        def callback(value):
            self.calls.append(value)
            self.loop.stop()

        self.loop.call_soon(callback, "called")
        self.loop.run_forever()
        self.assertEqual(self.calls, ["called"])

    def test_call_later(self):
        """Test call_later schedules a delayed callback."""
        start = time.monotonic()
        delay = 0.05

        def callback():
            elapsed = time.monotonic() - start
            self.calls.append(elapsed)
            self.loop.stop()

        self.loop.call_later(delay, callback)
        self.loop.run_forever()

        self.assertEqual(len(self.calls), 1)
        self.assertGreaterEqual(self.calls[0], delay - 0.01)  # Allow some tolerance

    def test_call_at(self):
        """Test call_at schedules a callback at absolute time."""
        start = self.loop.time()
        when = start + 0.05

        def callback():
            self.calls.append(self.loop.time())
            self.loop.stop()

        self.loop.call_at(when, callback)
        self.loop.run_forever()

        self.assertEqual(len(self.calls), 1)
        self.assertGreaterEqual(self.calls[0], when - 0.01)

    def test_handle_cancel(self):
        """Test cancelling a scheduled callback."""

        def callback():
            self.calls.append("called")

        handle = self.loop.call_soon(callback)
        handle.cancel()
        self.assertTrue(handle.cancelled())

        # Run a bit to ensure the cancelled callback doesn't run
        self.loop.call_soon(self.loop.stop)
        self.loop.run_forever()

        self.assertEqual(self.calls, [])

    def test_call_soon_threadsafe(self):
        """Test thread-safe callback scheduling."""
        result = []

        def background():
            time.sleep(0.01)
            self.loop.call_soon_threadsafe(result.append, "threaded")
            time.sleep(0.01)
            self.loop.call_soon_threadsafe(self.loop.stop)

        thread = threading.Thread(target=background)
        thread.start()

        self.loop.run_forever()
        thread.join()

        self.assertEqual(result, ["threaded"])

    def test_multiple_callbacks_order(self):
        """Test that callbacks run in order."""
        for i in range(5):
            self.loop.call_soon(lambda x=i: self.calls.append(x))
        self.loop.call_soon(self.loop.stop)

        self.loop.run_forever()
        self.assertEqual(self.calls, [0, 1, 2, 3, 4])

    def test_timer_handles_order(self):
        """Test that timer handles run in time order."""
        now = self.loop.time()

        self.loop.call_at(now + 0.03, lambda: self.calls.append(3))
        self.loop.call_at(now + 0.01, lambda: self.calls.append(1))
        self.loop.call_at(now + 0.02, lambda: self.calls.append(2))
        self.loop.call_at(now + 0.04, self.loop.stop)

        self.loop.run_forever()
        self.assertEqual(self.calls, [1, 2, 3])


class TestFuturesAndTasks(unittest.TestCase):
    """Tests for Future and Task handling."""

    def test_create_future(self):
        """Test creating a Future."""
        loop = new_event_loop()
        try:
            future = loop.create_future()
            self.assertIsInstance(future, asyncio.Future)
            self.assertFalse(future.done())
        finally:
            loop.close()

    def test_create_task(self):
        """Test creating a Task."""

        async def coro():
            return 42

        loop = new_event_loop()
        try:
            # Need to set the loop as running to create tasks
            async def wrapper():
                task = loop.create_task(coro())
                return await task

            result = loop.run_until_complete(wrapper())
            self.assertEqual(result, 42)
        finally:
            loop.close()

    def test_task_with_name(self):
        """Test creating a named task."""

        async def coro():
            return 42

        async def main():
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro(), name="my_task")
            self.assertEqual(task.get_name(), "my_task")
            return await task

        result = run(main())
        self.assertEqual(result, 42)

    def test_gather(self):
        """Test asyncio.gather works with CSP loop."""

        async def coro(n):
            await asyncio.sleep(0.01)
            return n * 2

        async def main():
            results = await asyncio.gather(coro(1), coro(2), coro(3))
            return results

        results = run(main())
        self.assertEqual(results, [2, 4, 6])

    def test_wait_for_timeout(self):
        """Test asyncio.wait_for with timeout."""

        async def slow_coro():
            await asyncio.sleep(10)
            return "done"

        async def main():
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(slow_coro(), timeout=0.01)

        run(main())

    def test_shield(self):
        """Test asyncio.shield."""

        async def inner():
            await asyncio.sleep(0.01)
            return "shielded"

        async def main():
            return await asyncio.shield(inner())

        result = run(main())
        self.assertEqual(result, "shielded")


class TestExceptionHandling(unittest.TestCase):
    """Tests for exception handling."""

    def test_exception_in_callback(self):
        """Test exception handling in callbacks."""
        loop = new_event_loop()
        exceptions = []

        def handler(loop, context):
            exceptions.append(context.get("exception"))

        loop.set_exception_handler(handler)

        def bad_callback():
            raise ValueError("test error")

        def stopper():
            loop.stop()

        try:
            loop.call_soon(bad_callback)
            loop.call_soon(stopper)
            loop.run_forever()

            self.assertEqual(len(exceptions), 1)
            self.assertIsInstance(exceptions[0], ValueError)
        finally:
            loop.close()

    def test_exception_in_coroutine(self):
        """Test exception handling in coroutines."""

        async def bad_coro():
            raise RuntimeError("test error")

        loop = new_event_loop()
        try:
            with self.assertRaises(RuntimeError):
                loop.run_until_complete(bad_coro())
        finally:
            loop.close()

    def test_default_exception_handler(self):
        """Test default exception handler is called."""
        loop = new_event_loop()

        with mock.patch.object(loop, "default_exception_handler") as mock_handler:

            def bad_callback():
                raise ValueError("test")

            def stopper():
                loop.stop()

            try:
                loop.call_soon(bad_callback)
                loop.call_soon(stopper)
                loop.run_forever()

                mock_handler.assert_called_once()
            finally:
                loop.close()


class TestIOOperations(unittest.TestCase):
    """Tests for I/O operations."""

    def test_add_remove_reader(self):
        """Test adding and removing file descriptor readers."""
        loop = new_event_loop()
        try:
            r, w = socket.socketpair()
            r.setblocking(False)
            w.setblocking(False)

            calls = []

            def reader():
                data = r.recv(1024)
                calls.append(data)
                loop.stop()

            loop.add_reader(r.fileno(), reader)
            w.send(b"test")
            loop.run_forever()

            self.assertEqual(calls, [b"test"])

            # Test remove
            result = loop.remove_reader(r.fileno())
            self.assertTrue(result)

            result = loop.remove_reader(r.fileno())
            self.assertFalse(result)

            r.close()
            w.close()
        finally:
            loop.close()

    def test_add_remove_writer(self):
        """Test adding and removing file descriptor writers."""
        loop = new_event_loop()
        try:
            r, w = socket.socketpair()
            r.setblocking(False)
            w.setblocking(False)

            calls = []

            def writer():
                w.send(b"test")
                calls.append(True)
                loop.remove_writer(w.fileno())
                loop.stop()

            loop.add_writer(w.fileno(), writer)
            loop.run_forever()

            self.assertEqual(calls, [True])
            self.assertEqual(r.recv(1024), b"test")

            r.close()
            w.close()
        finally:
            loop.close()


class TestExecutor(unittest.TestCase):
    """Tests for executor operations."""

    def test_run_in_executor(self):
        """Test running functions in executor."""

        def blocking_func(x, y):
            time.sleep(0.01)
            return x + y

        async def main():
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, blocking_func, 1, 2)
            return result

        result = run(main())
        self.assertEqual(result, 3)

    def test_run_in_custom_executor(self):
        """Test running with custom executor."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

            def blocking_func():
                return threading.current_thread().name

            async def main():
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, blocking_func)
                return result

            result = run(main())
            self.assertIn("ThreadPoolExecutor", result)


class TestEventLoopPolicy(unittest.TestCase):
    """Tests for CspEventLoopPolicy."""

    def test_policy_new_event_loop(self):
        """Test policy creates CspEventLoop."""
        policy = CspEventLoopPolicy()
        loop = policy.new_event_loop()
        self.assertIsInstance(loop, CspEventLoop)
        loop.close()

    def test_policy_get_set_event_loop(self):
        """Test policy get/set event loop."""
        policy = CspEventLoopPolicy()

        with self.assertRaises(RuntimeError):
            policy.get_event_loop()

        loop = policy.new_event_loop()
        policy.set_event_loop(loop)

        self.assertIs(policy.get_event_loop(), loop)

        policy.set_event_loop(None)
        loop.close()

    def test_set_as_global_policy(self):
        """Test setting as global asyncio policy."""
        old_policy = asyncio.get_event_loop_policy()
        try:
            asyncio.set_event_loop_policy(CspEventLoopPolicy())
            loop = asyncio.new_event_loop()
            self.assertIsInstance(loop, CspEventLoop)
            loop.close()
        finally:
            asyncio.set_event_loop_policy(old_policy)


class TestAsyncioCompatibility(unittest.TestCase):
    """Tests for asyncio compatibility."""

    def test_asyncio_sleep(self):
        """Test asyncio.sleep works."""

        async def main():
            start = time.monotonic()
            await asyncio.sleep(0.05)
            elapsed = time.monotonic() - start
            return elapsed

        result = run(main())
        self.assertGreaterEqual(result, 0.04)

    def test_asyncio_create_task(self):
        """Test asyncio.create_task works."""

        async def inner():
            await asyncio.sleep(0.01)
            return 42

        async def main():
            task = asyncio.create_task(inner())
            result = await task
            return result

        result = run(main())
        self.assertEqual(result, 42)

    def test_asyncio_wait(self):
        """Test asyncio.wait works."""

        async def coro(n):
            await asyncio.sleep(0.01)
            return n

        async def main():
            tasks = [asyncio.create_task(coro(i)) for i in range(3)]
            done, pending = await asyncio.wait(tasks)
            return sorted([t.result() for t in done])

        result = run(main())
        self.assertEqual(result, [0, 1, 2])

    def test_asyncio_queue(self):
        """Test asyncio.Queue works."""

        async def main():
            queue = asyncio.Queue()
            await queue.put(1)
            await queue.put(2)

            results = []
            results.append(await queue.get())
            results.append(await queue.get())
            return results

        result = run(main())
        self.assertEqual(result, [1, 2])

    def test_asyncio_event(self):
        """Test asyncio.Event works."""

        async def main():
            event = asyncio.Event()

            async def setter():
                await asyncio.sleep(0.01)
                event.set()

            asyncio.create_task(setter())
            await event.wait()
            return event.is_set()

        result = run(main())
        self.assertTrue(result)

    def test_asyncio_lock(self):
        """Test asyncio.Lock works."""

        async def main():
            lock = asyncio.Lock()
            results = []

            async def worker(n):
                async with lock:
                    results.append(f"start-{n}")
                    await asyncio.sleep(0.01)
                    results.append(f"end-{n}")

            await asyncio.gather(worker(1), worker(2))
            return results

        result = run(main())
        # Due to lock, operations should be sequential
        self.assertEqual(len(result), 4)
        # First worker should complete before second starts
        start_indices = [i for i, x in enumerate(result) if x.startswith("start")]
        end_indices = [i for i, x in enumerate(result) if x.startswith("end")]
        self.assertLess(end_indices[0], start_indices[1])

    def test_asyncio_semaphore(self):
        """Test asyncio.Semaphore works."""

        async def main():
            sem = asyncio.Semaphore(2)
            active = []
            max_active = [0]

            async def worker(n):
                async with sem:
                    active.append(n)
                    max_active[0] = max(max_active[0], len(active))
                    await asyncio.sleep(0.01)
                    active.remove(n)

            await asyncio.gather(*[worker(i) for i in range(5)])
            return max_active[0]

        result = run(main())
        self.assertLessEqual(result, 2)


class TestDebugMode(unittest.TestCase):
    """Tests for debug mode."""

    def test_debug_mode_default(self):
        """Test debug mode is off by default."""
        loop = new_event_loop()
        try:
            self.assertFalse(loop.get_debug())
        finally:
            loop.close()

    def test_set_debug_mode(self):
        """Test setting debug mode."""
        loop = new_event_loop()
        try:
            loop.set_debug(True)
            self.assertTrue(loop.get_debug())
            loop.set_debug(False)
            self.assertFalse(loop.get_debug())
        finally:
            loop.close()

    def test_run_with_debug(self):
        """Test running with debug mode."""

        async def main():
            loop = asyncio.get_running_loop()
            return loop.get_debug()

        result = run(main(), debug=True)
        self.assertTrue(result)


class TestLoopTime(unittest.TestCase):
    """Tests for loop time operations."""

    def test_time_increases(self):
        """Test that loop time increases."""
        loop = new_event_loop()
        try:
            t1 = loop.time()
            time.sleep(0.01)
            t2 = loop.time()
            self.assertGreater(t2, t1)
        finally:
            loop.close()


class TestContextVars(unittest.TestCase):
    """Tests for context variable support."""

    def test_context_in_callback(self):
        """Test context variables in callbacks."""
        import contextvars

        cv = contextvars.ContextVar("test_cv", default="default")
        results = []

        def callback():
            results.append(cv.get())
            loop.stop()

        loop = new_event_loop()
        try:
            ctx = contextvars.copy_context()
            ctx.run(cv.set, "modified")
            loop.call_soon(callback, context=ctx)
            loop.run_forever()

            self.assertEqual(results, ["modified"])
        finally:
            loop.close()


class TestShutdown(unittest.TestCase):
    """Tests for shutdown operations."""

    def test_shutdown_asyncgens(self):
        """Test shutdown_asyncgens."""
        cleanup_called = []

        async def async_gen():
            try:
                while True:
                    yield 1
                    await asyncio.sleep(0.01)
            finally:
                cleanup_called.append(True)

        async def main():
            loop = asyncio.get_running_loop()
            gen = async_gen()
            await gen.__anext__()
            # Don't complete the generator
            return loop

        # Run and capture the loop
        loop = new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            async def wrapper():
                gen = async_gen()
                await gen.__anext__()

            loop.run_until_complete(wrapper())
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
