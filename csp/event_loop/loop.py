import asyncio
import concurrent.futures
import contextvars
import os
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union

from csp.impl.__cspimpl import _cspimpl

_T = TypeVar("_T")

__all__ = (
    "CspEventLoop",
    "CspEventLoopPolicy",
    "new_event_loop",
    "run",
)


class _CspHandle:
    """Handle for a scheduled callback, compatible with asyncio.Handle."""

    __slots__ = ("_callback", "_args", "_context", "_loop", "_cancelled", "_repr")

    def __init__(
        self,
        callback: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        loop: Optional["CspEventLoop"] = None,
        context: Optional[contextvars.Context] = None,
    ):
        self._callback = callback
        self._args = args if args else ()
        self._context = context
        self._loop = loop
        self._cancelled = False
        self._repr = None

    def cancel(self) -> None:
        """Cancel the callback."""
        if not self._cancelled:
            self._cancelled = True
            self._callback = None
            self._args = None

    def cancelled(self) -> bool:
        """Return True if the callback was cancelled."""
        return self._cancelled

    def _run(self) -> None:
        """Execute the callback."""
        if self._cancelled:
            return
        try:
            if self._context is not None:
                self._context.run(self._callback, *self._args)
            else:
                self._callback(*self._args)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            if self._loop is not None:
                self._loop.call_exception_handler(
                    {
                        "message": "Exception in callback",
                        "exception": exc,
                        "handle": self,
                    }
                )
            else:
                raise

    def __repr__(self) -> str:
        if self._repr is None:
            info = [self.__class__.__name__]
            if self._cancelled:
                info.append("cancelled")
            if self._callback is not None:
                info.append(f"callback={self._callback!r}")
            self._repr = f"<{' '.join(info)}>"
        return self._repr


class _CspTimerHandle(_CspHandle):
    """Handle for a scheduled timer callback."""

    __slots__ = ("_when",)

    def __init__(
        self,
        when: float,
        callback: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        loop: Optional["CspEventLoop"] = None,
        context: Optional[contextvars.Context] = None,
    ):
        super().__init__(callback, args, loop, context)
        self._when = when

    def when(self) -> float:
        """Return the scheduled time as a float."""
        return self._when

    def __lt__(self, other: "_CspTimerHandle") -> bool:
        return self._when < other._when

    def __le__(self, other: "_CspTimerHandle") -> bool:
        return self._when <= other._when

    def __gt__(self, other: "_CspTimerHandle") -> bool:
        return self._when > other._when

    def __ge__(self, other: "_CspTimerHandle") -> bool:
        return self._when >= other._when

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CspTimerHandle):
            return self._when == other._when and self._callback == other._callback
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._when, id(self._callback)))


class CspEventLoop(asyncio.AbstractEventLoop):
    """
    An asyncio-compatible event loop backed by CSP's scheduler.

    This event loop integrates with CSP's realtime event processing capabilities,
    allowing asyncio coroutines to be scheduled alongside CSP graph computations.
    """

    def __init__(self, realtime: bool = True):
        """
        Initialize the CSP event loop.

        Args:
            realtime: If True, run in realtime mode (wall clock time).
                     If False, run in simulation mode.
        """
        self._realtime = realtime
        self._closed = False
        self._running = False
        self._stopping = False
        self._thread_id: Optional[int] = None
        self._debug = bool(os.environ.get("PYTHONASYNCIODEBUG"))

        # Callback queues
        self._ready: deque = deque()
        self._scheduled: List[_CspTimerHandle] = []

        # Selector for I/O
        self._selector = selectors.DefaultSelector()
        self._readers: Dict[int, _CspHandle] = {}
        self._writers: Dict[int, _CspHandle] = {}

        # Signal handling
        self._signal_handlers: Dict[int, _CspHandle] = {}
        self._ssock: Optional[socket.socket] = None
        self._csock: Optional[socket.socket] = None

        # Task factory and exception handler
        self._task_factory: Optional[Callable] = None
        self._exception_handler: Optional[Callable] = None
        self._default_executor: Optional[concurrent.futures.Executor] = None

        # For asyncgens
        self._asyncgens: set = set()
        self._asyncgens_shutdown_called = False

        # Time tracking
        self._clock_resolution = time.get_clock_info("monotonic").resolution
        self._start_time = time.monotonic()

        # CSP engine - now actively used for scheduling
        self._csp_engine: Optional[_cspimpl.PyEngine] = None
        self._csp_active = False
        self._starttime: Optional[datetime] = None
        self._endtime: Optional[datetime] = None
        self._sim_start_time: Optional[datetime] = None  # Track sim start for time()

        # Threadsafe callback queue
        self._csock_lock = threading.Lock()
        self._threadsafe_callbacks: deque = deque()

        # CSP wakeup fd for native event loop integration
        self._csp_wakeup_fd: Optional[int] = None

    def _init_csp_engine(self) -> None:
        """Initialize the CSP engine for use."""
        if self._csp_engine is None:
            self._csp_engine = _cspimpl.PyEngine(realtime=self._realtime)

    def _start_csp_engine(self) -> None:
        """Start CSP engine."""
        if not self._csp_active:
            self._init_csp_engine()
            from datetime import timedelta

            from csp.utils.datetime import utc_now

            if self._realtime:
                # Realtime mode: use current wall-clock time
                start = self._starttime or utc_now()
                end = self._endtime or (start + timedelta(days=365 * 100))
            else:
                # Simulation mode: use configured times or defaults
                # Default to Unix epoch for simulation if not specified
                start = self._starttime or datetime(1970, 1, 1)
                end = self._endtime or (start + timedelta(days=365 * 100))

            self._csp_engine.start(start, end)
            self._csp_active = True
            self._sim_start_time = start  # Track simulation start for time()

            # Register wakeup fd with selector for native event integration
            self._csp_wakeup_fd = self._csp_engine.get_wakeup_fd()
            if self._csp_wakeup_fd >= 0:
                try:
                    self._selector.register(self._csp_wakeup_fd, selectors.EVENT_READ)
                except (ValueError, OSError):
                    # Fd already registered or invalid, fall back to polling
                    self._csp_wakeup_fd = None

    def _stop_csp_engine(self) -> None:
        """Stop CSP engine."""
        if self._csp_active and self._csp_engine is not None:
            # Unregister wakeup fd from selector
            if self._csp_wakeup_fd is not None:
                try:
                    self._selector.unregister(self._csp_wakeup_fd)
                except (ValueError, OSError, KeyError):
                    pass  # Already unregistered or invalid
                self._csp_wakeup_fd = None
            try:
                self._csp_engine.finish()
            except Exception:
                pass  # Ignore errors during cleanup
            self._csp_active = False

    def set_simulation_time_range(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> None:
        """Configure the time range for simulation mode.

        This must be called before run_forever() or run_until_complete()
        to take effect.

        Args:
            start: Start time for simulation. If None, defaults to Unix epoch.
            end: End time for simulation. If None, defaults to 100 years from start.

        Raises:
            RuntimeError: If called on a realtime event loop.
            RuntimeError: If called while the loop is running.
        """
        if self._realtime:
            raise RuntimeError("Cannot set simulation time range on a realtime event loop")
        if self._running:
            raise RuntimeError("Cannot set simulation time range while the loop is running")
        self._starttime = start
        self._endtime = end

    def _check_closed(self) -> None:
        """Raise RuntimeError if the loop is closed."""
        if self._closed:
            raise RuntimeError("Event loop is closed")

    def _check_running(self) -> None:
        """Raise RuntimeError if the loop is running."""
        if self._running:
            raise RuntimeError("This event loop is already running")
        if asyncio._get_running_loop() is not None:
            raise RuntimeError("Cannot run the event loop while another loop is running")

    def _check_thread(self) -> None:
        """Raise RuntimeError if called from wrong thread."""
        if self._thread_id is not None and self._thread_id != threading.get_ident():
            raise RuntimeError("Non-thread-safe operation invoked on an event loop other than the current one")

    def run_forever(self) -> None:
        """Run the event loop until stop() is called."""
        self._check_closed()
        self._check_running()

        self._running = True
        self._thread_id = threading.get_ident()

        old_loop = asyncio._get_running_loop()
        try:
            asyncio._set_running_loop(self)
            self._start_csp_engine()
            self._run_until_stopped()
        finally:
            self._stop_csp_engine()
            asyncio._set_running_loop(old_loop)
            self._running = False
            self._thread_id = None

    def _run_until_stopped(self) -> None:
        """Internal implementation of run_forever."""
        while not self._stopping:
            self._run_once()
        self._stopping = False

    def _run_once(self, timeout: Optional[float] = None) -> None:
        """Run one iteration of the event loop.

        This integrates CSP's scheduler with asyncio's I/O handling.
        CSP handles timing/scheduling while selectors handle I/O.

        In realtime mode:
          - Uses wall-clock time
          - Waits on selectors for I/O events
          - CSP processes push events from external sources

        In simulation mode:
          - CSP's scheduler drives time progression
          - No waiting - events are processed as fast as possible
          - Time jumps instantly to next scheduled event
        """
        # Process threadsafe callbacks first
        self._process_threadsafe_callbacks()

        if self._realtime:
            # Realtime mode: use wall-clock time and wait on I/O
            self._run_once_realtime(timeout)
        else:
            # Simulation mode: CSP drives time, no waiting
            self._run_once_simulation()

    def _run_once_realtime(self, timeout: Optional[float] = None) -> None:
        """Run one iteration in realtime mode."""

        # Calculate timeout for selector
        if self._ready:
            # Have ready callbacks, don't wait
            timeout = 0
        elif self._scheduled:
            # Calculate based on next scheduled Python callback
            when = self._scheduled[0]._when
            timeout = max(0, when - self.time())
        else:
            timeout = timeout if timeout is not None else 1.0

        # Also check CSP's next scheduled time if active
        if self._csp_active and self._csp_engine is not None:
            csp_next = self._csp_engine.next_scheduled_time()
            if csp_next is not None:
                csp_now = self._csp_engine.now()
                if csp_now is not None:
                    csp_wait = (csp_next - csp_now).total_seconds()
                    if csp_wait >= 0:
                        timeout = min(timeout, csp_wait)

        # Poll for I/O events
        try:
            events = self._selector.select(timeout)
        except (OSError, ValueError):
            events = []

        # Track if CSP wakeup fd was signaled
        csp_wakeup_signaled = False

        # Process I/O events
        for key, mask in events:
            # Check if this is the CSP wakeup fd
            if self._csp_wakeup_fd is not None and key.fd == self._csp_wakeup_fd:
                csp_wakeup_signaled = True
                continue  # Don't add to readers dict
            if mask & selectors.EVENT_READ and key.fd in self._readers:
                self._ready.append(self._readers[key.fd])
            if mask & selectors.EVENT_WRITE and key.fd in self._writers:
                self._ready.append(self._writers[key.fd])

        # Step CSP engine if active and (wakeup signaled OR scheduled events due)
        if self._csp_active and self._csp_engine is not None:
            if csp_wakeup_signaled:
                # Clear the wakeup fd before processing
                self._csp_engine.clear_wakeup_fd()
            try:
                # Step with 0 wait - just process what's ready
                self._csp_engine.process_one_cycle(0.0)
            except Exception:
                pass  # CSP cycle errors handled elsewhere

        # Process Python scheduled callbacks that are due
        now = self.time()
        while self._scheduled:
            handle = self._scheduled[0]
            if handle._when > now:
                break
            handle = self._scheduled.pop(0)
            if not handle.cancelled():
                self._ready.append(handle)

        # Run ready callbacks
        ntodo = len(self._ready)
        for _ in range(ntodo):
            handle = self._ready.popleft()
            if not handle.cancelled():
                handle._run()

    def _run_once_simulation(self) -> None:
        """Run one iteration in simulation mode.

        In simulation mode:
        - Time jumps instantly to the next scheduled event
        - No waiting on selectors (just poll for ready I/O)
        - CSP's scheduler drives the time progression
        """

        # Poll for I/O events without waiting (timeout=0)
        try:
            events = self._selector.select(0)
        except (OSError, ValueError):
            events = []

        # Process I/O events
        for key, mask in events:
            if mask & selectors.EVENT_READ and key.fd in self._readers:
                self._ready.append(self._readers[key.fd])
            if mask & selectors.EVENT_WRITE and key.fd in self._writers:
                self._ready.append(self._writers[key.fd])

        # Step CSP engine - this advances simulated time to next event
        if self._csp_active and self._csp_engine is not None:
            try:
                # Step with 0 wait - in sim mode this jumps to next event
                self._csp_engine.process_one_cycle(0.0)
            except Exception:
                pass  # CSP cycle errors handled elsewhere

        # Process Python scheduled callbacks that are due
        # In sim mode, time() returns CSP's simulated time
        now = self.time()
        while self._scheduled:
            handle = self._scheduled[0]
            if handle._when > now:
                break
            handle = self._scheduled.pop(0)
            if not handle.cancelled():
                self._ready.append(handle)

        # Run ready callbacks
        ntodo = len(self._ready)
        for _ in range(ntodo):
            handle = self._ready.popleft()
            if not handle.cancelled():
                handle._run()

    def _process_threadsafe_callbacks(self) -> None:
        """Process callbacks added via call_soon_threadsafe."""
        while True:
            try:
                handle = self._threadsafe_callbacks.popleft()
                self._ready.append(handle)
            except IndexError:
                break

    def run_until_complete(self, future: Union[asyncio.Future, Coroutine]) -> Any:
        """Run until the future is complete."""
        self._check_closed()
        self._check_running()

        new_task = not asyncio.isfuture(future)
        future = asyncio.ensure_future(future, loop=self)
        if new_task:
            future._log_destroy_pending = False

        def done_callback(fut: asyncio.Future) -> None:
            if not fut.cancelled():
                exc = fut.exception()
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    return
            self.stop()

        future.add_done_callback(done_callback)

        try:
            self.run_forever()
        except BaseException:
            if new_task and future.done() and not future.cancelled():
                future.exception()
            raise
        finally:
            future.remove_done_callback(done_callback)

        if not future.done():
            raise RuntimeError("Event loop stopped before Future completed.")

        return future.result()

    def stop(self) -> None:
        """Stop the event loop."""
        self._stopping = True

    def is_running(self) -> bool:
        """Return True if the loop is running."""
        return self._running

    def is_closed(self) -> bool:
        """Return True if the loop is closed."""
        return self._closed

    def close(self) -> None:
        """Close the event loop."""
        if self._running:
            raise RuntimeError("Cannot close a running event loop")
        if self._closed:
            return

        self._closed = True

        # Stop CSP engine if active
        self._stop_csp_engine()
        self._csp_engine = None

        # Clear callbacks
        self._ready.clear()
        self._scheduled.clear()

        # Close selector
        self._selector.close()

        # Shutdown default executor
        if self._default_executor is not None:
            self._default_executor.shutdown(wait=False)
            self._default_executor = None

    async def shutdown_asyncgens(self) -> None:
        """Shutdown all active asynchronous generators."""
        self._asyncgens_shutdown_called = True

        if not self._asyncgens:
            return

        closing_agens = list(self._asyncgens)
        self._asyncgens.clear()

        results = await asyncio.gather(*[ag.aclose() for ag in closing_agens], return_exceptions=True)

        for result, agen in zip(results, closing_agens):
            if isinstance(result, Exception):
                self.call_exception_handler(
                    {
                        "message": f"an error occurred during closing of asynchronous generator {agen!r}",
                        "exception": result,
                        "asyncgen": agen,
                    }
                )

    async def shutdown_default_executor(self, timeout: Optional[float] = None) -> None:
        """Schedule the shutdown of the default executor."""
        if self._default_executor is None:
            return

        executor = self._default_executor
        self._default_executor = None

        def shutdown_executor():
            executor.shutdown(wait=True)

        # Create a new single-use executor to run the shutdown
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as temp_executor:
            future = self.create_future()

            def callback(result: concurrent.futures.Future) -> None:
                if future.cancelled():
                    return
                try:
                    result.result()
                except Exception as exc:
                    self.call_soon_threadsafe(future.set_exception, exc)
                else:
                    self.call_soon_threadsafe(future.set_result, None)

            temp_executor.submit(shutdown_executor).add_done_callback(callback)

            if timeout is not None:
                try:
                    await asyncio.wait_for(future, timeout)
                except asyncio.TimeoutError:
                    warnings.warn(
                        f"The default executor did not finish shutting down in {timeout} seconds",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                await future

    def call_soon(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspHandle:
        """Schedule a callback to be called as soon as possible."""
        self._check_closed()
        if self._debug:
            self._check_thread()

        handle = _CspHandle(callback, args if args else None, self, context)
        self._ready.append(handle)
        return handle

    def call_soon_threadsafe(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspHandle:
        """Thread-safe version of call_soon."""
        self._check_closed()

        handle = _CspHandle(callback, args if args else None, self, context)
        self._threadsafe_callbacks.append(handle)

        # Wake up the event loop if needed
        with self._csock_lock:
            if self._csock is not None:
                try:
                    self._csock.send(b"\x00")
                except OSError:
                    pass

        return handle

    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspTimerHandle:
        """Schedule a callback to be called after delay seconds."""
        self._check_closed()
        if self._debug:
            self._check_thread()

        if delay < 0:
            delay = 0

        when = self.time() + delay
        return self.call_at(when, callback, *args, context=context)

    def call_at(
        self,
        when: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspTimerHandle:
        """Schedule a callback to be called at absolute time when."""
        self._check_closed()
        if self._debug:
            self._check_thread()

        handle = _CspTimerHandle(when, callback, args if args else None, self, context)

        # Insert in sorted order
        # Using binary search would be more efficient for large lists
        import bisect

        bisect.insort(self._scheduled, handle)

        return handle

    def time(self) -> float:
        """Return the current time according to the event loop's clock.

        In realtime mode, returns the monotonic clock time.
        In simulation mode, returns CSP's simulated time as seconds since epoch.
        """
        if not self._realtime and self._csp_active and self._csp_engine is not None:
            # Simulation mode: use CSP's simulated time
            csp_now = self._csp_engine.now()
            if csp_now is not None:
                # Convert datetime to float seconds since start
                return csp_now.timestamp()
        # Realtime mode or no CSP engine: use monotonic clock
        return time.monotonic()

    def create_future(self) -> asyncio.Future:
        """Create and return a new Future."""
        return asyncio.Future(loop=self)

    def create_task(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: Optional[str] = None,
        context: Optional[contextvars.Context] = None,
    ) -> asyncio.Task[_T]:
        """Schedule a coroutine to run as a Task."""
        self._check_closed()

        if self._task_factory is None:
            # context parameter was added in Python 3.11
            if sys.version_info >= (3, 11):
                task = asyncio.Task(coro, loop=self, name=name, context=context)
            else:
                task = asyncio.Task(coro, loop=self, name=name)
        else:
            if context is None:
                task = self._task_factory(self, coro)
            else:
                task = context.run(self._task_factory, self, coro)
            if name is not None and hasattr(task, "set_name"):
                task.set_name(name)

        return task

    def set_task_factory(self, factory: Optional[Callable]) -> None:
        """Set a task factory."""
        if factory is not None and not callable(factory):
            raise TypeError("task factory must be a callable or None")
        self._task_factory = factory

    def get_task_factory(self) -> Optional[Callable]:
        """Get the current task factory."""
        return self._task_factory

    def add_reader(self, fd: int, callback: Callable[..., Any], *args: Any) -> None:
        """Add a reader callback for a file descriptor."""
        self._check_closed()
        handle = _CspHandle(callback, args if args else None, self)

        if fd in self._readers:
            self.remove_reader(fd)

        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, selectors.EVENT_READ, None)
        else:
            mask = key.events | selectors.EVENT_READ
            self._selector.modify(fd, mask, None)

        self._readers[fd] = handle

    def remove_reader(self, fd: int) -> bool:
        """Remove a reader callback for a file descriptor."""
        if fd not in self._readers:
            return False

        del self._readers[fd]

        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return True

        if key.events & selectors.EVENT_WRITE:
            self._selector.modify(fd, selectors.EVENT_WRITE, None)
        else:
            self._selector.unregister(fd)

        return True

    def add_writer(self, fd: int, callback: Callable[..., Any], *args: Any) -> None:
        """Add a writer callback for a file descriptor."""
        self._check_closed()
        handle = _CspHandle(callback, args if args else None, self)

        if fd in self._writers:
            self.remove_writer(fd)

        try:
            key = self._selector.get_key(fd)
        except KeyError:
            self._selector.register(fd, selectors.EVENT_WRITE, None)
        else:
            mask = key.events | selectors.EVENT_WRITE
            self._selector.modify(fd, mask, None)

        self._writers[fd] = handle

    def remove_writer(self, fd: int) -> bool:
        """Remove a writer callback for a file descriptor."""
        if fd not in self._writers:
            return False

        del self._writers[fd]

        try:
            key = self._selector.get_key(fd)
        except KeyError:
            return True

        if key.events & selectors.EVENT_READ:
            self._selector.modify(fd, selectors.EVENT_READ, None)
        else:
            self._selector.unregister(fd)

        return True

    async def sock_recv(self, sock: socket.socket, nbytes: int) -> bytes:
        """Receive data from a socket."""
        fut = self.create_future()
        fd = sock.fileno()

        def callback() -> None:
            try:
                data = sock.recv(nbytes)
            except (BlockingIOError, InterruptedError):
                return  # Try again
            except Exception as exc:
                self.remove_reader(fd)
                fut.set_exception(exc)
            else:
                self.remove_reader(fd)
                fut.set_result(data)

        self.add_reader(fd, callback)
        return await fut

    async def sock_recv_into(self, sock: socket.socket, buf: bytearray) -> int:
        """Receive data from a socket into a buffer."""
        fut = self.create_future()
        fd = sock.fileno()

        def callback() -> None:
            try:
                nbytes = sock.recv_into(buf)
            except (BlockingIOError, InterruptedError):
                return  # Try again
            except Exception as exc:
                self.remove_reader(fd)
                fut.set_exception(exc)
            else:
                self.remove_reader(fd)
                fut.set_result(nbytes)

        self.add_reader(fd, callback)
        return await fut

    async def sock_sendall(self, sock: socket.socket, data: bytes) -> None:
        """Send data to a socket."""
        fut = self.create_future()
        fd = sock.fileno()
        view = memoryview(data)

        def callback() -> None:
            nonlocal view
            try:
                n = sock.send(view)
            except (BlockingIOError, InterruptedError):
                return  # Try again
            except Exception as exc:
                self.remove_writer(fd)
                fut.set_exception(exc)
                return

            if n == len(view):
                self.remove_writer(fd)
                fut.set_result(None)
            else:
                view = view[n:]

        self.add_writer(fd, callback)
        await fut

    async def sock_connect(self, sock: socket.socket, address: Tuple[str, int]) -> None:
        """Connect a socket to a remote address."""
        try:
            sock.connect(address)
            return
        except (BlockingIOError, InterruptedError):
            pass

        fut = self.create_future()
        fd = sock.fileno()

        def callback() -> None:
            try:
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if err != 0:
                    raise OSError(err, f"Connect call failed {address}")
            except Exception as exc:
                self.remove_writer(fd)
                fut.set_exception(exc)
            else:
                self.remove_writer(fd)
                fut.set_result(None)

        self.add_writer(fd, callback)
        await fut

    async def sock_accept(self, sock: socket.socket) -> Tuple[socket.socket, Tuple[str, int]]:
        """Accept a connection on a socket."""
        fut = self.create_future()
        fd = sock.fileno()

        def callback() -> None:
            try:
                conn, addr = sock.accept()
                conn.setblocking(False)
            except (BlockingIOError, InterruptedError):
                return  # Try again
            except Exception as exc:
                self.remove_reader(fd)
                fut.set_exception(exc)
            else:
                self.remove_reader(fd)
                fut.set_result((conn, addr))

        self.add_reader(fd, callback)
        return await fut

    async def getaddrinfo(
        self,
        host: Optional[str],
        port: Optional[Union[str, int]],
        *,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> List[Tuple[int, int, int, str, Tuple[str, int]]]:
        """Look up address info for a host."""
        return await self.run_in_executor(
            None,
            socket.getaddrinfo,
            host,
            port,
            family,
            type,
            proto,
            flags,
        )

    async def getnameinfo(self, sockaddr: Tuple[str, int], flags: int = 0) -> Tuple[str, str]:
        """Look up name info for an address."""
        return await self.run_in_executor(None, socket.getnameinfo, sockaddr, flags)

    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        func: Callable[..., _T],
        *args: Any,
    ) -> asyncio.Future[_T]:
        """Run a function in an executor."""
        self._check_closed()

        if executor is None:
            if self._default_executor is None:
                self._default_executor = concurrent.futures.ThreadPoolExecutor()
            executor = self._default_executor

        fut = asyncio.Future(loop=self)

        def callback(result: concurrent.futures.Future) -> None:
            if fut.cancelled():
                return
            try:
                res = result.result()
            except Exception as exc:
                self.call_soon_threadsafe(fut.set_exception, exc)
            else:
                self.call_soon_threadsafe(fut.set_result, res)

        executor.submit(func, *args).add_done_callback(callback)
        return fut

    def set_default_executor(self, executor: Optional[concurrent.futures.Executor]) -> None:
        """Set the default executor."""
        if not isinstance(executor, (type(None), concurrent.futures.Executor)):
            raise TypeError("executor must be Executor instance or None")
        self._default_executor = executor

    def set_exception_handler(self, handler: Optional[Callable]) -> None:
        """Set an exception handler."""
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable or None")
        self._exception_handler = handler

    def get_exception_handler(self) -> Optional[Callable]:
        """Get the current exception handler."""
        return self._exception_handler

    def default_exception_handler(self, context: Dict[str, Any]) -> None:
        """Default exception handler."""
        message = context.get("message")
        if not message:
            message = "Unhandled exception in event loop"

        exception = context.get("exception")

        log_lines = [message]
        for key, value in sorted(context.items()):
            if key in {"message", "exception"}:
                continue
            log_lines.append(f"{key}: {value!r}")

        if exception is not None:
            import traceback

            exc_info = (type(exception), exception, exception.__traceback__)
            log_lines.append("Exception:")
            log_lines.extend(traceback.format_exception(*exc_info))

        print("\n".join(log_lines), file=sys.stderr)

    def call_exception_handler(self, context: Dict[str, Any]) -> None:
        """Call the exception handler."""
        if self._exception_handler is None:
            try:
                self.default_exception_handler(context)
            except Exception:
                print("Exception in default exception handler:", file=sys.stderr)
                import traceback

                traceback.print_exc()
        else:
            try:
                self._exception_handler(self, context)
            except Exception as exc:
                try:
                    self.default_exception_handler(
                        {
                            "message": "Exception in exception handler",
                            "exception": exc,
                            "context": context,
                        }
                    )
                except Exception:
                    print("Exception in exception handler:", file=sys.stderr)
                    import traceback

                    traceback.print_exc()

    def get_debug(self) -> bool:
        """Return True if debug mode is enabled."""
        return self._debug

    def set_debug(self, enabled: bool) -> None:
        """Set debug mode."""
        self._debug = enabled

    async def subprocess_exec(
        self,
        protocol_factory: Callable[[], asyncio.SubprocessProtocol],
        *args: Any,
        stdin: Any = subprocess.PIPE,
        stdout: Any = subprocess.PIPE,
        stderr: Any = subprocess.PIPE,
        **kwargs: Any,
    ) -> Tuple[asyncio.SubprocessTransport, asyncio.SubprocessProtocol]:
        """Execute a subprocess."""
        raise NotImplementedError("subprocess_exec is not yet implemented for CSP event loop")

    async def subprocess_shell(
        self,
        protocol_factory: Callable[[], asyncio.SubprocessProtocol],
        cmd: str,
        *,
        stdin: Any = subprocess.PIPE,
        stdout: Any = subprocess.PIPE,
        stderr: Any = subprocess.PIPE,
        **kwargs: Any,
    ) -> Tuple[asyncio.SubprocessTransport, asyncio.SubprocessProtocol]:
        """Execute a shell command as a subprocess."""
        raise NotImplementedError("subprocess_shell is not yet implemented for CSP event loop")

    def add_signal_handler(self, sig: int, callback: Callable[..., Any], *args: Any) -> None:
        """Add a handler for a signal."""
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("Signal handlers can only be set in the main thread")

        self._check_closed()
        handle = _CspHandle(callback, args if args else None, self)
        self._signal_handlers[sig] = handle

        try:
            signal.signal(sig, lambda s, f: self.call_soon_threadsafe(callback, *args))
        except OSError:
            del self._signal_handlers[sig]
            raise

    def remove_signal_handler(self, sig: int) -> bool:
        """Remove a handler for a signal."""
        if sig not in self._signal_handlers:
            return False

        del self._signal_handlers[sig]

        try:
            signal.signal(sig, signal.SIG_DFL)
        except OSError:
            pass

        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} running={self._running} closed={self._closed} debug={self._debug}>"


class CspEventLoopPolicy(asyncio.AbstractEventLoopPolicy):
    """
    Event loop policy for CSP-backed asyncio.

    This policy creates CspEventLoop instances for asyncio operations.
    """

    class _Local(threading.local):
        _loop: Optional[CspEventLoop] = None

    def __init__(self) -> None:
        self._local = self._Local()

    def get_event_loop(self) -> CspEventLoop:
        """Get the event loop for the current context."""
        if self._local._loop is None:
            raise RuntimeError(f"There is no current event loop in thread {threading.current_thread().name!r}.")
        return self._local._loop

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        """Set the event loop for the current context."""
        if loop is not None and not isinstance(loop, asyncio.AbstractEventLoop):
            raise TypeError(f"loop must be an instance of AbstractEventLoop or None, not '{type(loop).__name__}'")
        self._local._loop = loop

    def new_event_loop(self) -> CspEventLoop:
        """Create and return a new event loop."""
        return CspEventLoop()


def new_event_loop() -> CspEventLoop:
    """Create and return a new CSP event loop."""
    return CspEventLoop()


def run(
    main: Coroutine[Any, Any, _T],
    *,
    loop_factory: Optional[Callable[[], CspEventLoop]] = None,
    debug: Optional[bool] = None,
) -> _T:
    """
    Run a coroutine using the CSP event loop.

    This is the preferred way to run asyncio code with the CSP event loop.

    Args:
        main: The coroutine to run
        loop_factory: Optional factory function to create the event loop.
                     Defaults to new_event_loop.
        debug: If True, run in debug mode.

    Returns:
        The result of the coroutine.

    Example:
        async def main():
            await asyncio.sleep(1)
            return "done"

        result = csp.event_loop.run(main())
    """
    if loop_factory is None:
        loop_factory = new_event_loop

    if asyncio._get_running_loop() is not None:
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    if not asyncio.iscoroutine(main):
        raise ValueError(f"a coroutine was expected, got {main!r}")

    loop = loop_factory()
    try:
        asyncio.set_event_loop(loop)
        if debug is not None:
            loop.set_debug(debug)
        return loop.run_until_complete(main)
    finally:
        try:
            _cancel_all_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            if hasattr(loop, "shutdown_default_executor"):
                loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all pending tasks."""
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )
