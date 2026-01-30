import asyncio
import atexit
import concurrent.futures
import queue
import threading
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import csp
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.types.tstype import ts
from csp.impl.wiring import py_push_adapter_def

__all__ = [
    "async_for",
    "async_in",
    "async_out",
    "async_node",
    "await_",
    "async_alarm",
    "schedule_async_alarm",
    "get_async_loop",
    "get_shared_loop",
    "get_csp_asyncio_loop",
    "is_csp_asyncio_mode",
    "shutdown_shared_loop",
]

T = TypeVar("T")
U = TypeVar("U")


_shared_loop: Optional[asyncio.AbstractEventLoop] = None
_shared_thread: Optional[threading.Thread] = None
_shared_lock = threading.Lock()
_shared_ready = threading.Event()


def get_running_loop_or_none() -> Optional[asyncio.AbstractEventLoop]:
    """
    Get the currently running asyncio event loop, or None if not in an async context.

    This is used to detect if we're inside a CspEventLoop or other asyncio loop,
    in which case we can schedule async operations directly without a background thread.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def get_csp_asyncio_loop() -> Optional[asyncio.AbstractEventLoop]:
    """
    Get the asyncio event loop from CSP's asyncio mode, if enabled.

    Returns:
        The asyncio loop if CSP is running in asyncio mode (realtime=True with
        asyncio_on_thread=False, the default), else None.
    """
    from csp.impl.wiring.runtime import GraphRunInfo

    try:
        info = GraphRunInfo.get_cur_run_times_info(raise_if_missing=False)
        if info is not None and info.is_asyncio:
            return info.asyncio_loop
    except Exception:
        pass
    return None


def is_csp_asyncio_mode() -> bool:
    """
    Check if CSP is currently running in asyncio mode.

    Returns:
        True if CSP is running in asyncio mode (realtime=True with
        asyncio_on_thread=False, the default), else False.
    """
    return get_csp_asyncio_loop() is not None


def get_async_loop() -> asyncio.AbstractEventLoop:
    """
    Get the appropriate asyncio event loop for running async operations.

    Priority:
    1. If CSP is running in asyncio mode (realtime with asyncio_on_thread=False), use that loop
    2. If there's a running asyncio loop (e.g., CspEventLoop), use it directly
    3. Otherwise, use/create the shared background loop

    This allows async adapters to integrate directly with CSP's asyncio mode
    or CspEventLoop when available, avoiding the overhead of a separate background thread.

    Returns:
        An asyncio event loop suitable for scheduling coroutines.
    """
    # Check if CSP is running in asyncio mode
    csp_loop = get_csp_asyncio_loop()
    if csp_loop is not None:
        return csp_loop

    # Check if we're already inside an asyncio context (e.g., CspEventLoop)
    running_loop = get_running_loop_or_none()
    if running_loop is not None:
        return running_loop

    # Fall back to shared background loop
    return get_shared_loop()


def get_shared_loop() -> asyncio.AbstractEventLoop:
    """
    Get the shared asyncio event loop for running async operations.

    This returns a lazily-initialized event loop running in a background thread.
    All async adapters can reuse this loop instead of creating their own threads.

    The loop is automatically shut down when the process exits.

    Returns:
        The shared asyncio event loop.
    """
    global _shared_loop, _shared_thread

    if _shared_loop is not None and _shared_loop.is_running():
        return _shared_loop

    with _shared_lock:
        # Double-check after acquiring lock
        if _shared_loop is not None and _shared_loop.is_running():
            return _shared_loop

        _shared_ready.clear()

        def run_loop():
            global _shared_loop
            _shared_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_shared_loop)
            _shared_ready.set()
            try:
                _shared_loop.run_forever()
            finally:
                try:
                    _shared_loop.close()
                except Exception:
                    pass

        _shared_thread = threading.Thread(target=run_loop, daemon=True, name="csp-async-loop")
        _shared_thread.start()
        _shared_ready.wait(timeout=5.0)

        if _shared_loop is None:
            raise RuntimeError("Failed to start shared async loop")

        return _shared_loop


def shutdown_shared_loop() -> None:
    """
    Shut down the shared async loop.

    This is called automatically at process exit, but can be called manually
    if you need to cleanly shut down before exit.
    """
    global _shared_loop, _shared_thread

    with _shared_lock:
        if _shared_loop is not None and _shared_loop.is_running():
            _shared_loop.call_soon_threadsafe(_shared_loop.stop)

        if _shared_thread is not None:
            _shared_thread.join(timeout=2.0)
            _shared_thread = None

        _shared_loop = None


# Register cleanup at exit
atexit.register(shutdown_shared_loop)


def _run_on_async_loop(coro: Awaitable[T], timeout: Optional[float] = None) -> T:
    """
    Run a coroutine on the best available loop and wait for the result.

    If we're inside a running asyncio loop (e.g., CspEventLoop), uses that directly.
    Otherwise, uses the shared background loop.

    Args:
        coro: The coroutine to run.
        timeout: Optional timeout in seconds.

    Returns:
        The result of the coroutine.
    """
    # Always use the shared background loop for blocking calls.
    # Even if there's a running asyncio loop (e.g., CSP's asyncio mode),
    # we can't run a blocking call on it — that would deadlock the event
    # loop.  The shared loop runs in a separate thread, so
    # future.result() safely blocks the calling thread while the
    # coroutine progresses on the background thread.
    loop = get_shared_loop()
    future = concurrent.futures.Future()

    async def wrapper():
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout)
            else:
                result = await coro
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(wrapper(), loop=loop))
    return future.result(timeout=timeout)


def _schedule_on_loop(loop: asyncio.AbstractEventLoop, callback, *args):
    """
    Schedule a callback on the event loop, handling both same-thread and cross-thread cases.

    If called from within the loop's thread and the loop is running, uses call_soon.
    Otherwise uses call_soon_threadsafe.
    """
    try:
        running_loop = asyncio.get_running_loop()
        if running_loop is loop:
            # We're on the same thread as the loop, use call_soon
            loop.call_soon(callback, *args)
            return
    except RuntimeError:
        pass

    # Cross-thread or no running loop, use threadsafe version
    loop.call_soon_threadsafe(callback, *args)


def _schedule_coro_on_loop(loop: asyncio.AbstractEventLoop, coro) -> None:
    """
    Schedule a coroutine on the event loop, handling both same-thread and cross-thread cases.

    Args:
        loop: The event loop to schedule on.
        coro: The coroutine to schedule.
    """

    def schedule():
        asyncio.ensure_future(coro, loop=loop)

    _schedule_on_loop(loop, schedule)


def _schedule_on_async_loop(coro: Awaitable[T]) -> concurrent.futures.Future:
    """
    Schedule a coroutine on the best available loop without waiting.

    If we're inside a running asyncio loop (e.g., CspEventLoop), schedules there.
    Otherwise, uses the shared background loop.

    Args:
        coro: The coroutine to run.

    Returns:
        A Future that will contain the result.
    """
    loop = get_async_loop()
    future = concurrent.futures.Future()

    async def wrapper():
        try:
            result = await coro
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    _schedule_coro_on_loop(loop, wrapper())
    return future


def _extract_async_iterator_type(type_hint) -> type:
    """
    Extract the element type from an AsyncIterator[T] or AsyncGenerator[T, ...] type hint.

    For example:
        AsyncIterator[int] -> int
        AsyncGenerator[str, None] -> str
    """
    origin = get_origin(type_hint)
    if origin is not None:
        # Check if it's AsyncIterator, AsyncGenerator, or similar
        args = get_args(type_hint)
        if args:
            return args[0]  # First type argument is the yield type
    return type_hint


class _AsyncForAdapterImpl(PushInputAdapter):
    """Push adapter implementation that consumes an async generator and pushes values to CSP."""

    def __init__(
        self,
        async_gen: AsyncIterator,
        output_type: type,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._async_gen = async_gen
        self._output_type = output_type
        self._provided_loop = loop
        self._thread: threading.Thread = None
        self._loop: asyncio.AbstractEventLoop = None
        self._active = False
        self._task: asyncio.Task = None

    def start(self, starttime, endtime):
        self._active = True
        if self._provided_loop is not None:
            # Use the provided loop
            self._loop = self._provided_loop
        else:
            # Use the best available loop (running loop or shared loop)
            self._loop = get_async_loop()
        _schedule_on_loop(self._loop, self._schedule_consumer)

    def _schedule_consumer(self):
        """Schedule the consumer coroutine on the shared loop."""
        self._task = asyncio.ensure_future(self._consume_generator(), loop=self._loop)

    def stop(self):
        self._active = False
        # Cancel the task on the loop
        if self._task is not None and not self._task.done():
            _schedule_on_loop(self._loop, self._task.cancel)

    async def _consume_generator(self):
        """Consume the async generator and push each value to CSP."""
        try:
            async for value in self._async_gen:
                if not self._active:
                    break
                self.push_tick(value)
        except asyncio.CancelledError:
            pass


_AsyncForAdapter = py_push_adapter_def(
    "AsyncForAdapter",
    _AsyncForAdapterImpl,
    ts["T"],
    async_gen=object,
    output_type="T",
    loop=object,
)


def async_for(
    async_gen_or_func: AsyncIterator[T],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]:
    """
    Bridge an async generator to CSP, creating a time series that ticks on each yielded value.

    The async generator function must have a return type annotation specifying the output type.

    Args:
        async_gen_or_func: An async generator instance (result of calling an async generator function).
        loop: Event loop to use for running async operations. If None (default), uses CSP's
              shared async loop which is efficient as all adapters share one background thread.

    Returns:
        A CSP time series (ts[T]) that ticks whenever the async generator yields a value.

    Example:
        async def my_async_gen(n: int) -> AsyncIterator[int]:
            for i in range(n):
                await asyncio.sleep(0.1)
                yield i

        @csp.graph
        def my_graph():
            values = csp.async_for(my_async_gen(10))
            csp.print("value", values)
    """
    # Get the output type from the async generator
    if hasattr(async_gen_or_func, "ag_frame"):
        # It's an async generator instance - get the function from the code object
        ag_code = async_gen_or_func.ag_code
        # Try to get type hints from the frame's globals
        func_name = ag_code.co_name
        func_globals = async_gen_or_func.ag_frame.f_globals

        # Look for the function in globals to get type hints
        if func_name in func_globals:
            func = func_globals[func_name]
            try:
                hints = get_type_hints(func)
                return_hint = hints.get("return", object)
                # Extract the element type from AsyncIterator[T] or similar
                output_type = _extract_async_iterator_type(return_hint)
            except Exception:
                output_type = object
        else:
            output_type = object
    else:
        raise TypeError(
            "async_for expects an async generator instance. "
            "Make sure to call the async generator function, e.g., async_for(my_gen(args)) not async_for(my_gen)"
        )

    return _AsyncForAdapter(async_gen_or_func, output_type, loop)


class _AsyncInAdapterImpl(PushInputAdapter):
    """Push adapter that runs a coroutine and pushes the result when ready."""

    def __init__(
        self,
        coro: Coroutine,
        output_type: type,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._coro = coro
        self._output_type = output_type
        self._provided_loop = loop
        self._loop: asyncio.AbstractEventLoop = None

    def start(self, starttime, endtime):
        # Use provided loop, running loop, or shared loop
        self._loop = self._provided_loop if self._provided_loop is not None else get_async_loop()

        async def run_and_push():
            try:
                result = await self._coro
                self.push_tick(result)
            except Exception:
                pass

        _schedule_coro_on_loop(self._loop, run_and_push())

    def stop(self):
        pass  # Nothing to clean up - loop is managed externally


_AsyncInAdapter = py_push_adapter_def(
    "AsyncInAdapter",
    _AsyncInAdapterImpl,
    ts["T"],
    coro=object,
    output_type="T",
    loop=object,
)


def async_in(
    coro: Awaitable[T],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]:
    """
    Run an async coroutine and create a time series that ticks once when it completes.

    Args:
        coro: A coroutine instance (result of calling an async function).
        loop: Event loop to use for running async operations. If None (default), uses CSP's
              shared async loop which is efficient as all adapters share one background thread.

    Returns:
        A CSP time series (ts[T]) that ticks once with the coroutine's return value.

    Example:
        async def fetch_data() -> int:
            await asyncio.sleep(0.1)
            return 42

        @csp.graph
        def my_graph():
            value = csp.async_in(fetch_data())
            csp.print("value", value)
    """
    # Get output type from the coroutine
    if hasattr(coro, "cr_code"):
        # It's a coroutine - get type hints from the function
        func_name = coro.cr_code.co_name
        func_globals = coro.cr_frame.f_globals if coro.cr_frame else {}

        if func_name in func_globals:
            func = func_globals[func_name]
            try:
                hints = get_type_hints(func)
                output_type = hints.get("return", object)
            except Exception:
                output_type = object
        else:
            output_type = object
    else:
        raise TypeError(
            "async_in expects a coroutine instance. "
            "Make sure to call the async function, e.g., async_in(my_func()) not async_in(my_func)"
        )

    return _AsyncInAdapter(coro, output_type, loop)


@csp.node
def async_out(
    x: ts["T"],
    async_func: Callable[["T"], Awaitable[None]],
    loop: object = None,  # Optional[asyncio.AbstractEventLoop], but object for csp.node compatibility
):
    """
    Invoke an async function whenever the input time series ticks.

    Args:
        x: Input time series that triggers the async function.
        async_func: An async function that takes the ticked value. Should return None.
        loop: Event loop to use for running async operations. If None (default), uses CSP's
              shared async loop which is efficient as all adapters share one background thread.

    Example:
        async def send_data(n: int) -> None:
            await asyncio.sleep(0.1)
            print(f"Sent: {n}")

        @csp.graph
        def my_graph():
            values = ...  # some ts[int]
            csp.async_out(values, send_data)
    """
    with csp.state():
        s_loop = None

    with csp.start():
        # Use provided loop, running loop, or shared loop
        s_loop = loop if loop is not None else get_async_loop()

    if csp.ticked(x):
        if s_loop is not None:
            # Schedule directly on the loop
            _schedule_coro_on_loop(s_loop, async_func(x))


class _AsyncNodeState:
    """Shared state for the async node pattern."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.provided_loop = loop
        self.loop: asyncio.AbstractEventLoop = None
        self.active = True
        self.input_queue: asyncio.Queue = None
        self.push_adapter = None


class _AsyncNodeOutputAdapterImpl(PushInputAdapter):
    """Push adapter for async node output."""

    def __init__(self, state: _AsyncNodeState, async_func: Callable, output_type: type):
        self._state = state
        self._async_func = async_func
        self._output_type = output_type
        self._tasks = []

    def start(self, starttime, endtime):
        self._state.push_adapter = self

        # Use provided loop, running loop, or shared loop
        self._state.loop = self._state.provided_loop if self._state.provided_loop is not None else get_async_loop()
        self._state.input_queue = asyncio.Queue()
        # Schedule the processor on the loop
        _schedule_on_loop(self._state.loop, self._schedule_processor)

    def _schedule_processor(self):
        """Schedule the async processor on the shared loop."""

        async def process_one(value):
            try:
                result = await self._async_func(value)
                self.push_tick(result)
            except Exception:
                pass

        async def queue_processor():
            while self._state.active:
                try:
                    value = await asyncio.wait_for(self._state.input_queue.get(), timeout=0.1)
                    # Process each value as a separate task for concurrency
                    task = asyncio.ensure_future(process_one(value), loop=self._state.loop)
                    self._tasks.append(task)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

        self._processor_task = asyncio.ensure_future(queue_processor(), loop=self._state.loop)

    def stop(self):
        self._state.active = False
        # Cancel the processor task
        if hasattr(self, "_processor_task") and not self._processor_task.done():
            _schedule_on_loop(self._state.loop, self._processor_task.cancel)
        # Cancel pending tasks
        for task in self._tasks:
            if not task.done():
                _schedule_on_loop(self._state.loop, task.cancel)


_AsyncNodeOutputAdapter = py_push_adapter_def(
    "AsyncNodeOutputAdapter",
    _AsyncNodeOutputAdapterImpl,
    ts["T"],
    state=object,
    async_func=object,
    output_type="T",
)


@csp.node
def _async_node_input(x: ts["T"], state: _AsyncNodeState):
    """Helper node that feeds input values to the async processing queue."""
    if csp.ticked(x):
        if state.loop is not None and state.loop.is_running():
            _schedule_on_loop(state.loop, state.input_queue.put_nowait, x)


def async_node(
    x: ts["T"],
    async_func: Callable[["T"], Awaitable["U"]],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts["U"]:
    """
    Apply an async function to each tick of the input, outputting the results.

    Takes a CSP input, runs an async function on each value, and outputs
    the results as a new time series.

    Args:
        x: Input time series.
        async_func: An async function that transforms the input value.
        loop: Event loop to use for running async operations. If None (default), uses CSP's
              shared async loop which is efficient as all adapters share one background thread.

    Returns:
        A CSP time series with the async function's results.

    Example:
        async def process(n: int) -> int:
            await asyncio.sleep(0.1)
            return n * 2

        @csp.graph
        def my_graph():
            values = ...  # some ts[int]
            results = csp.async_node(values, process)
            csp.print("results", results)
    """
    # Get output type from the async function
    try:
        hints = get_type_hints(async_func)
        output_type = hints.get("return", object)
    except Exception:
        output_type = object

    state = _AsyncNodeState(loop)

    # Wire up the input feeder and output adapter
    _async_node_input(x, state)
    return _AsyncNodeOutputAdapter(state, async_func, output_type)


def await_(
    coro: Awaitable[T],
    block: bool = True,
    timeout: float = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> T:
    """
    Await an async coroutine from synchronous code.

    This function allows calling async code from within CSP nodes or other
    synchronous contexts.

    Args:
        coro: A coroutine instance (result of calling an async function).
        block: If True (default), blocks until the coroutine completes.
               If False, returns a Future that can be checked later.
        timeout: Optional timeout in seconds.
        loop: Event loop to use. If None (default), uses CSP's shared loop
              which is more efficient as it reuses a single background thread.
              Pass your own loop for custom behavior.

    Returns:
        When block=True: The result of the coroutine.
        When block=False: A Future object that will contain the result.

    Example:
        @csp.node
        def my_node(x: ts[int]) -> ts[int]:
            if csp.ticked(x):
                # Uses CSP's shared async loop (efficient, default)
                result = csp.await_(async_func(x))
                return result

        # Or with a custom loop:
        result = csp.await_(async_func(x), loop=my_custom_loop)
    """
    if loop is None:
        # Use the best available loop
        if block:
            return _run_on_async_loop(coro, timeout)
        else:
            return _schedule_on_async_loop(coro)
    else:
        # Use the provided loop
        if block:
            future = concurrent.futures.Future()

            async def wrapper():
                try:
                    if timeout is not None:
                        result = await asyncio.wait_for(coro, timeout)
                    else:
                        result = await coro
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            _schedule_coro_on_loop(loop, wrapper())
            return future.result(timeout=timeout)
        else:
            future = concurrent.futures.Future()

            async def wrapper():
                try:
                    result = await coro
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            _schedule_coro_on_loop(loop, wrapper())
            return future


class AsyncContext:
    """
    Context manager for managing async operations within CSP nodes.

    Provides a shared event loop and thread for running async operations,
    avoiding the overhead of creating new loops for each operation.

    Example:
        @csp.node
        def my_node(x: ts[int]) -> ts[int]:
            with csp.state():
                s_ctx = None

            with csp.start():
                s_ctx = csp.AsyncContext()
                s_ctx.start()

            with csp.stop():
                if s_ctx:
                    s_ctx.stop()

            if csp.ticked(x):
                result = s_ctx.run(async_func(x))
                return result
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._active = False
        self._ready = threading.Event()

    def start(self):
        """Start the async context's event loop in a background thread."""
        if self._active:
            return

        self._active = True

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        """Stop the async context's event loop."""
        self._active = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def run(self, coro: Awaitable[T], timeout: float = None) -> T:
        """
        Run a coroutine in this context's event loop and wait for result.

        Args:
            coro: A coroutine to run.
            timeout: Optional timeout in seconds.

        Returns:
            The result of the coroutine.
        """
        if not self._ready.is_set():
            raise RuntimeError("AsyncContext not started. Call start() first.")

        import concurrent.futures

        future = concurrent.futures.Future()

        async def wrapper():
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(coro, timeout)
                else:
                    result = await coro
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        _schedule_coro_on_loop(self._loop, wrapper())
        return future.result(timeout=timeout)

    def run_nowait(self, coro: Awaitable[T]) -> "concurrent.futures.Future[T]":
        """
        Schedule a coroutine to run without waiting for the result.

        Args:
            coro: A coroutine to run.

        Returns:
            A Future that will contain the result when complete.
        """
        import concurrent.futures

        if not self._ready.is_set():
            raise RuntimeError("AsyncContext not started. Call start() first.")

        future = concurrent.futures.Future()

        async def wrapper():
            try:
                result = await coro
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        _schedule_coro_on_loop(self._loop, wrapper())
        return future

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class AsyncAlarm:
    """
    Async alarm that fires when async operations complete.

    This class provides an alarm-like interface for async operations within CSP nodes.
    When an async operation completes, the alarm "ticks" with the result value.

    The internal mechanism uses a background thread with an event loop to run async
    operations, and a polling alarm to check for completed results.

    Example:
        @csp.node
        def my_node() -> ts[int]:
            with csp.alarms():
                poll_alarm = csp.alarm(bool)
                async_alarm = csp.async_alarm(int)

            with csp.state():
                s_counter = 0
                s_pending = False

            with csp.start():
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(poll_alarm):
                # Only schedule a new async operation if one isn't already pending
                if not s_pending:
                    s_counter += 1
                    csp.schedule_async_alarm(async_alarm, async_func(s_counter))
                    s_pending = True

                # Keep polling
                csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

            if csp.ticked(async_alarm):
                # Async operation completed - we can schedule another one now
                s_pending = False
                return async_alarm
    """

    def __init__(self, output_type: type = object):
        self._output_type = output_type
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._active = False
        self._ready = threading.Event()
        self._results: queue.Queue = queue.Queue()
        self._pending_count = 0
        self._lock = threading.Lock()
        self._last_result = None  # Store the last result for value access

    def start(self):
        """Start the async alarm's event loop in a background thread."""
        if self._active:
            return

        self._active = True

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        """Stop the async alarm's event loop and cancel pending tasks."""
        self._active = False
        if self._loop is not None and self._loop.is_running():
            # Cancel all pending tasks before stopping
            def cancel_all():
                for task in asyncio.all_tasks(self._loop):
                    task.cancel()
                self._loop.stop()

            self._loop.call_soon_threadsafe(cancel_all)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def schedule(self, coro: Awaitable[T]) -> None:
        """
        Schedule an async operation. When it completes, the result will be available.

        Args:
            coro: A coroutine to run.
        """
        if not self._ready.is_set():
            raise RuntimeError("AsyncAlarm not started. Call start() first.")

        with self._lock:
            self._pending_count += 1

        async def wrapper():
            try:
                result = await coro
                self._results.put(("success", result))
            except Exception as e:
                self._results.put(("error", e))
            finally:
                with self._lock:
                    self._pending_count -= 1

        _schedule_coro_on_loop(self._loop, wrapper())

    def has_result(self) -> bool:
        """Check if any async operation has completed and has a result waiting."""
        return not self._results.empty()

    def get_result(self) -> T:
        """
        Get the next completed result. Also stores it as _last_result for value access.

        Returns:
            The result of the completed async operation.

        Raises:
            queue.Empty: If no result is available.
            Exception: If the async operation raised an exception.
        """
        try:
            status, value = self._results.get_nowait()
            if status == "error":
                raise value
            self._last_result = value
            return value
        except queue.Empty:
            raise

    @property
    def value(self) -> T:
        """Get the last result value. Used when accessing the alarm as a value."""
        return self._last_result

    def pending_count(self) -> int:
        """Return the number of pending async operations."""
        with self._lock:
            return self._pending_count


# Convenience functions for alarm-like syntax
def async_alarm(output_type: type = object) -> AsyncAlarm:
    """
    Create an async alarm for use in CSP nodes.

    This is meant to be used in a pattern similar to csp.alarm(), but for async operations.
    The async alarm is automatically started when created and should be stopped in the
    node's stop block.

    Args:
        output_type: The type of values that will be produced by async operations.

    Returns:
        An AsyncAlarm instance (already started).

    Example:
        with csp.alarms():
            async_alarm = csp.async_alarm(int)

        with csp.stop():
            async_alarm.stop()
    """
    alarm = AsyncAlarm(output_type)
    alarm.start()  # Auto-start for convenience
    return alarm


def schedule_async_alarm(alarm: AsyncAlarm, coro: Awaitable[T]) -> None:
    """
    Schedule an async operation on an async alarm.

    When the async operation completes, the alarm will have a result available.

    Args:
        alarm: The AsyncAlarm to schedule on.
        coro: The coroutine to run.

    Example:
        csp.schedule_async_alarm(s_async_alarm, fetch_data(url))
    """
    alarm.schedule(coro)
