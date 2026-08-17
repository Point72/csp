import asyncio
import atexit
import concurrent.futures
import logging
import queue
import threading
import warnings
from datetime import timedelta
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
]

T = TypeVar("T")
U = TypeVar("U")

log = logging.getLogger(__name__)


def _warn_type_fallback(api: str, func_name: str) -> None:
    """Element types are recovered by looking the function up in its own module globals, which
    misses methods, closures, decorated and shadowed names -- silently changing the resulting
    ts[] type."""
    warnings.warn(
        f"{api} could not infer an element type for {func_name!r} and fell back to `object`; "
        "the resulting ts[] will be ts[object]",
        stacklevel=3,
    )


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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _shared_loop = loop
            # Signal readiness from inside the loop. Setting it before run_forever() lets a
            # concurrent caller fail the is_running() re-check under the lock and start a second
            # thread, orphaning this one along with everything scheduled on it.
            loop.call_soon(_shared_ready.set)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        _shared_thread = threading.Thread(target=run_loop, daemon=True, name="csp-async-loop")
        _shared_thread.start()
        if not _shared_ready.wait(timeout=5.0):
            raise RuntimeError("Timed out waiting for the shared async loop to start")

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

    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_complete_future(future, coro, timeout), loop=loop))
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


async def _complete_future(future: "concurrent.futures.Future", coro, timeout: Optional[float] = None) -> None:
    """
    Await `coro` and complete `future` with its outcome.

    Catches BaseException, not Exception: asyncio.CancelledError derives from BaseException, so an
    Exception-only handler leaves the future uncompleted and any thread blocked on
    future.result() waiting forever.
    """
    try:
        if timeout is not None:
            result = await asyncio.wait_for(coro, timeout)
        else:
            result = await coro
        if not future.done():
            future.set_result(result)
    except BaseException as e:
        if not future.done():
            future.set_exception(e)
        if isinstance(e, asyncio.CancelledError):
            raise  # let asyncio mark the task cancelled rather than failed


def _cancel_task_sync(loop: asyncio.AbstractEventLoop, task, timeout: float = 5.0) -> None:
    """
    Cancel `task` and do not return until it can no longer run.

    csp adapter stop() is followed by engine teardown, so returning while the task can still
    resume means it can push into an adapter whose C++ backing is gone.

    When called from the loop's own thread (same-thread asyncio mode) blocking would deadlock, but
    it is also unnecessary: nothing else runs on this thread until we yield, so the cancellation is
    guaranteed to be seen before the task next resumes.
    """
    if task is None or task.done():
        return

    try:
        on_loop_thread = asyncio.get_running_loop() is loop
    except RuntimeError:
        on_loop_thread = False

    if on_loop_thread:
        task.cancel()
        return

    finished = threading.Event()

    def cancel():
        task.add_done_callback(lambda _: finished.set())
        task.cancel()

    try:
        _schedule_on_loop(loop, cancel)
    except RuntimeError:
        # Loop already closed, so the task can never resume
        return

    if not finished.wait(timeout):
        raise RuntimeError(f"timed out after {timeout}s waiting for async task {task!r} to cancel")


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

    _schedule_coro_on_loop(loop, _complete_future(future, coro))
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
        _cancel_task_sync(self._loop, self._task)
        self._task = None

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
    output_type: Optional[type] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]:
    """
    Bridge an async generator to CSP, creating a time series that ticks on each yielded value.

    The async generator function must have a return type annotation specifying the output type.

    Args:
        async_gen_or_func: An async generator instance (result of calling an async generator function).
        output_type: Element type of the resulting series. Pass this when the generator is a method,
              a closure or otherwise not resolvable from its module globals, where inference falls
              back to `object`.
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
        if output_type is None:
            # It's an async generator instance - get the function from the code object
            ag_code = async_gen_or_func.ag_code
            # Try to get type hints from the frame's globals
            func_name = ag_code.co_name
            func_globals = async_gen_or_func.ag_frame.f_globals

            # Look for the function in globals to get type hints
            output_type = object
            if func_name in func_globals:
                func = func_globals[func_name]
                try:
                    hints = get_type_hints(func)
                    return_hint = hints.get("return", object)
                    # Extract the element type from AsyncIterator[T] or similar
                    output_type = _extract_async_iterator_type(return_hint)
                except Exception:
                    output_type = object

            if output_type is object:
                _warn_type_fallback("async_for", func_name)
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
        self._active = False
        self._task: asyncio.Task = None

    def start(self, starttime, endtime):
        # Use provided loop, running loop, or shared loop
        self._active = True
        self._loop = self._provided_loop if self._provided_loop is not None else get_async_loop()
        _schedule_on_loop(self._loop, self._schedule_runner)

    def _schedule_runner(self):
        self._task = asyncio.ensure_future(self._run_and_push(), loop=self._loop)

    async def _run_and_push(self):
        try:
            result = await self._coro
            if self._active:
                self.push_tick(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The awaited coroutine is the whole point of this adapter; dropping its failure
            # leaves the graph waiting on a tick that will never come, with no diagnostic.
            self._loop.call_exception_handler({"message": "async_in coroutine failed", "exception": exc})

    def stop(self):
        # The coroutine can complete after teardown starts; returning before it is cancelled
        # would let it push into an adapter whose C++ backing is gone.
        self._active = False
        _cancel_task_sync(self._loop, self._task)
        self._task = None


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
    output_type: Optional[type] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]:
    """
    Run an async coroutine and create a time series that ticks once when it completes.

    Args:
        coro: A coroutine instance (result of calling an async function).
        output_type: Type of the resulting series. Pass this when the coroutine function is a
              method, a closure or otherwise not resolvable from its module globals, where
              inference falls back to `object`.
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
        if output_type is None:
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

            if output_type is object:
                _warn_type_fallback("async_in", func_name)
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

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        maxsize: int = 0,
        on_overflow: str = "drop_oldest",
    ):
        self.provided_loop = loop
        self.loop: asyncio.AbstractEventLoop = None
        self.active = True
        self.maxsize = maxsize
        self.on_overflow = on_overflow
        self.input_queue: asyncio.Queue = None
        self.push_adapter = None
        self.dropped = 0
        self.dropped_reported = False

    def enqueue(self, value) -> None:
        """Put a tick on the queue, applying the overflow policy. Runs on the loop thread."""
        queue_ = self.input_queue
        if queue_ is None or not self.active:
            return

        # maxsize 0 is unbounded, so full() is never true and neither branch below is taken
        if queue_.full():
            if self.on_overflow == "drop_newest":
                self.dropped += 1
                return
            # drop_oldest: make room by discarding the stalest item
            queue_.get_nowait()
            self.dropped += 1

        queue_.put_nowait(value)


# Pushed on stop so the queue processor wakes immediately instead of being polled awake
_ASYNC_NODE_SHUTDOWN = object()

_ASYNC_NODE_OVERFLOW_POLICIES = ("drop_oldest", "drop_newest")


class _AsyncNodeOutputAdapterImpl(PushInputAdapter):
    """Push adapter for async node output."""

    def __init__(self, state: _AsyncNodeState, async_func: Callable, output_type: type):
        self._state = state
        self._async_func = async_func
        self._output_type = output_type
        self._tasks = set()

    def start(self, starttime, endtime):
        self._state.push_adapter = self

        # Use provided loop, running loop, or shared loop
        self._state.loop = self._state.provided_loop if self._state.provided_loop is not None else get_async_loop()
        self._state.input_queue = asyncio.Queue(maxsize=self._state.maxsize)
        # Schedule the processor on the loop
        _schedule_on_loop(self._state.loop, self._schedule_processor)

    def _schedule_processor(self):
        """Schedule the async processor on the shared loop."""
        self._processor_task = asyncio.ensure_future(self._queue_processor(), loop=self._state.loop)

    async def _process_one(self, value, slot=None):
        try:
            result = await self._async_func(value)
            self.push_tick(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._report("async_node function failed", exc)
        finally:
            if slot is not None:
                slot.release()

    async def _queue_processor(self):
        # Bounding the queue alone would not bound anything: this loop drains it as fast as it
        # can into tasks, so without a concurrency limit the backlog just moves into _tasks.
        # Acquiring before dequeuing is what pushes back on the queue and triggers the drop
        # policy.  maxsize 0 means unbounded, which is the historical behaviour.
        slots = asyncio.Semaphore(self._state.maxsize) if self._state.maxsize else None

        while self._state.active:
            try:
                if slots is not None:
                    await slots.acquire()

                value = await self._state.input_queue.get()
                if value is _ASYNC_NODE_SHUTDOWN:
                    if slots is not None:
                        slots.release()
                    break

                # Process each value as a separate task for concurrency.  Hold a strong
                # reference until it completes - asyncio only weakly references tasks - and
                # drop it afterwards so this does not accumulate one Task per tick.
                task = asyncio.ensure_future(self._process_one(value, slots), loop=self._state.loop)
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                # Without a report a persistent failure here is an invisible hot loop
                if slots is not None:
                    slots.release()
                self._report("async_node queue processor failed", exc)

    def _report(self, message: str, exc: BaseException) -> None:
        loop = self._state.loop
        if loop is not None:
            loop.call_exception_handler({"message": message, "exception": exc})
        else:
            log.error(message, exc_info=exc)

    def stop(self):
        self._state.active = False
        queue_ = self._state.input_queue
        loop = self._state.loop
        if queue_ is not None and loop is not None and loop.is_running():
            # The graceful exit: the processor leaves its loop instead of unwinding through a
            # cancellation.  Cancellation below is still what guarantees termination, since in
            # same-thread asyncio mode the loop cannot reach this callback during teardown.
            # A full bounded queue has no room for the sentinel, so make some.
            def signal_shutdown():
                if queue_.full():
                    queue_.get_nowait()
                queue_.put_nowait(_ASYNC_NODE_SHUTDOWN)

            _schedule_on_loop(loop, signal_shutdown)

        if hasattr(self, "_processor_task"):
            _cancel_task_sync(self._state.loop, self._processor_task)
        for task in list(self._tasks):
            _cancel_task_sync(self._state.loop, task)
        self._tasks.clear()

        # Read the counter only once the loop has quiesced, so a drop racing with teardown is
        # still counted, and only once, so a repeated stop() does not repeat the warning
        if self._state.dropped and not self._state.dropped_reported:
            self._state.dropped_reported = True
            log.warning("async_node dropped %d input ticks that did not fit its queue", self._state.dropped)


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
            _schedule_on_loop(state.loop, state.enqueue, x)


def async_node(
    x: ts["T"],
    async_func: Callable[["T"], Awaitable["U"]],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    maxsize: int = 0,
    on_overflow: str = "drop_oldest",
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
        maxsize: Bound on outstanding work: at most this many ticks wait in the queue and at
                 most this many run concurrently. 0 (default) is unbounded, which means an
                 async_func slower than the input grows memory without limit; set a bound if
                 the input can outrun the function.
        on_overflow: What to do with a tick that arrives at a full queue, either "drop_oldest"
                     (default, keeps the freshest data) or "drop_newest" (keeps the accepted
                     sequence intact). Still validated but has no effect when maxsize is 0.
                     Drops are counted and logged once at stop.

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
    if maxsize < 0:
        raise ValueError(f"async_node maxsize must be >= 0, got {maxsize}")
    if on_overflow not in _ASYNC_NODE_OVERFLOW_POLICIES:
        raise ValueError(f"async_node on_overflow must be one of {_ASYNC_NODE_OVERFLOW_POLICIES}, got {on_overflow!r}")

    # Get output type from the async function
    try:
        hints = get_type_hints(async_func)
        output_type = hints.get("return", object)
    except Exception:
        output_type = object

    state = _AsyncNodeState(loop, maxsize=maxsize, on_overflow=on_overflow)

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
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if block:
            if running_loop is loop:
                coro.close()
                raise RuntimeError(
                    "await_(block=True, loop=...) cannot be called from that loop's own thread: "
                    "the coroutine cannot run until this call returns, so it would deadlock. "
                    "Use block=False, or await the coroutine directly."
                )

            future = concurrent.futures.Future()

            _schedule_coro_on_loop(loop, _complete_future(future, coro, timeout))
            return future.result(timeout=timeout)
        else:
            future = concurrent.futures.Future()

            _schedule_coro_on_loop(loop, _complete_future(future, coro))
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
                s_ctx = AsyncContext()
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            # Signal from inside the loop: setting _ready before run_forever() lets run() schedule
            # onto a loop that is not spinning yet.
            loop.call_soon(self._ready.set)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("timed out waiting for AsyncContext loop thread to start")

    def stop(self):
        """Stop the async context's event loop."""
        self._active = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("timed out waiting for AsyncContext loop thread to stop")

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

        future = concurrent.futures.Future()

        _schedule_coro_on_loop(self._loop, _complete_future(future, coro, timeout))
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

        _schedule_coro_on_loop(self._loop, _complete_future(future, coro))
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

    In same-thread asyncio mode the coroutine runs on the engine's own loop and its completion
    wakes the node directly, so no polling alarm is needed.  When there is no engine loop to
    borrow the alarm falls back to a background thread, and the node only observes the result the
    next time something else ticks it.

    Example:
        @csp.node
        def my_node(trigger: ts[object]) -> ts[int]:
            with csp.alarms():
                async_alarm = csp.async_alarm(int)

            with csp.state():
                s_counter = 0

            if csp.ticked(trigger):
                s_counter += 1
                csp.schedule_async_alarm(async_alarm, async_func(s_counter))

            if csp.ticked(async_alarm):
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
        self._last_error: Optional[BaseException] = None
        self._error_observed = False
        self._ticked_this_cycle = False
        self._wakeup_proxy = None
        self._owns_loop = False
        self._tasks = set()

    def bind_wakeup(self, proxy) -> None:
        """Bind this alarm's own input proxy, used to wake the node when a result lands."""
        self._wakeup_proxy = proxy

    def start(self):
        """Start the async alarm's event loop."""
        if self._active:
            return

        self._active = True

        # In same-thread asyncio mode the engine's own loop is available, and completions on it
        # run on the engine thread - which is what makes the alarm wakeup below legal.
        csp_loop = get_csp_asyncio_loop()
        if csp_loop is not None:
            self._loop = csp_loop
            self._owns_loop = False
            self._ready.set()
            return

        self._owns_loop = True

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            # Signal from inside the loop: setting _ready before run_forever() lets schedule()
            # hand work to a loop that is not spinning yet.
            loop.call_soon(self._ready.set)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("timed out waiting for AsyncAlarm loop thread to start")

    def stop(self):
        """Stop the async alarm's event loop and cancel pending tasks."""
        self._active = False
        self._wakeup_proxy = None

        # On a shared loop the tasks outlive this alarm unless cancelled here; stop() runs on the
        # loop's own thread in that mode, so cancel() is enough to keep them from resuming.
        for task in list(self._tasks):
            _cancel_task_sync(self._loop, task)
        self._tasks.clear()

        if not self._owns_loop:
            return
        if self._loop is not None and self._loop.is_running():
            # Cancel all pending tasks before stopping
            def cancel_all():
                for task in asyncio.all_tasks(self._loop):
                    task.cancel()
                self._loop.stop()

            self._loop.call_soon_threadsafe(cancel_all)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("timed out waiting for AsyncAlarm loop thread to stop")

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
                self._wake_node(result)
            except Exception as e:
                self._results.put(("error", e))
                # Wake for failures too, else a node with no other ticking input never raises
                self._wake_node(self._placeholder_value())
            finally:
                with self._lock:
                    self._pending_count -= 1

        def schedule():
            task = asyncio.ensure_future(wrapper(), loop=self._loop)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        _schedule_on_loop(self._loop, schedule)

    def _placeholder_value(self):
        """A value of the alarm's type to carry a failure wakeup, which never reaches the node."""
        try:
            return self._output_type()
        except Exception:
            return None

    def _wake_node(self, value) -> None:
        """Fire this alarm's own slot so the node runs now rather than on the user's next poll.

        Only legal when the completion ran on the engine thread, which is the case when the
        coroutine was scheduled on CSP's own loop. Otherwise the result waits for a poll.
        """
        proxy = self._wakeup_proxy
        if proxy is None or not self._active or self._owns_loop:
            return
        try:
            proxy.schedule_alarm(timedelta(0), value)
        except Exception as exc:
            # Without a wakeup a node with no other ticking input never sees this result, so this
            # must not pass silently
            self._loop.call_exception_handler({"message": "async alarm wakeup failed", "exception": exc})

    def has_result(self) -> bool:
        """Check if any async operation has completed and has a result waiting."""
        return self._ticked_this_cycle or not self._results.empty()

    def start_cycle(self) -> None:
        """Clear the per-cycle latch. Called once per node invocation by generated code."""
        # A node may tick on a failed alarm and never read it; without this the exception would
        # vanish, since csp.ticked() no longer raises
        if self._last_error is not None and not self._error_observed:
            self._error_observed = True
            log.error("async alarm coroutine failed and the node never read the value", exc_info=self._last_error)
        self._ticked_this_cycle = False

    def get_result(self) -> T:
        """
        Latch the result for this cycle, dequeuing at most once per node invocation.

        A failed coroutine is latched rather than raised here, because this runs inside
        csp.ticked(), which everywhere else in csp is a plain predicate. The exception is
        raised when the node reads the alarm's value.

        Raises:
            queue.Empty: If no result is available.
        """
        if self._ticked_this_cycle:
            return self._last_result

        status, value = self._results.get_nowait()
        self._ticked_this_cycle = True

        if status == "error":
            self._last_error = value
            self._error_observed = False
            self._last_result = None
            return None

        self._last_error = None
        self._last_result = value
        return value

    @property
    def value(self) -> T:
        """Get the last result value. Used when accessing the alarm as a value."""
        if self._last_error is not None:
            self._error_observed = True
            raise self._last_error
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
    The alarm is started when created and stopped for you when the node stops.

    Args:
        output_type: The type of values that will be produced by async operations.

    Returns:
        An AsyncAlarm instance (already started).

    Example:
        with csp.alarms():
            async_alarm = csp.async_alarm(int)
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
