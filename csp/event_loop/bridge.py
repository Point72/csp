import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, List, Optional, TypeVar

from csp.impl.genericpushadapter import GenericPushAdapter

_T = TypeVar("_T")

__all__ = ("AsyncioBridge", "BidirectionalBridge")


class AsyncioBridge:
    """
    Bridge between asyncio event loop and CSP's realtime engine.

    This class provides the ability to schedule callbacks that execute
    within a running CSP graph, interacting with csp.now() and CSP's time.

    The bridge runs its own asyncio event loop in a background thread,
    allowing you to schedule callbacks and run coroutines that push
    data to CSP via a GenericPushAdapter.

    Attributes:
        adapter: The GenericPushAdapter used to push data to CSP.

    Example:
        >>> bridge = AsyncioBridge(int, "counter")
        >>> bridge.start()
        >>>
        >>> # Schedule a callback
        >>> bridge.call_later(1.0, lambda: bridge.push(42))
        >>>
        >>> # Run a coroutine
        >>> async def fetch_and_push():
        ...     data = await some_async_operation()
        ...     bridge.push(data)
        >>> bridge.run_coroutine(fetch_and_push())
        >>>
        >>> bridge.stop()
    """

    def __init__(self, adapter_type: type = object, name: str = "asyncio_bridge"):
        """
        Initialize the asyncio bridge.

        Args:
            adapter_type: The type of data to push through the adapter.
                          This should match the type expected by your CSP nodes.
            name: Name for the push adapter (for debugging/identification).
        """
        self._adapter = GenericPushAdapter(adapter_type, name)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._start_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def adapter(self) -> GenericPushAdapter:
        """
        Get the underlying push adapter to wire into CSP graph.

        Use this in your graph definition to get the edge that receives
        data pushed via this bridge.

        Returns:
            The GenericPushAdapter instance.

        Example:
            >>> @csp.graph
            ... def my_graph():
            ...     data = bridge.adapter.out()
            ...     # data is now a ts[adapter_type] edge
        """
        return self._adapter

    @property
    def is_running(self) -> bool:
        """Check if the bridge is currently running."""
        return self._running

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the underlying asyncio event loop (if started)."""
        return self._loop

    def start(self, start_time: Optional[datetime] = None) -> None:
        """
        Start the asyncio event loop in a background thread.

        This must be called before scheduling callbacks or running coroutines.
        The bridge can be started before or after the CSP graph starts.

        Args:
            start_time: The CSP engine start time (used for time calculations
                        with call_at). If not provided, uses current UTC time.

        Raises:
            RuntimeError: If the bridge is already running.
        """
        if self._running:
            raise RuntimeError("Bridge is already running")

        self._start_time = start_time or datetime.utcnow()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        timeout = 5.0
        start_wait = time.time()
        while self._loop is None and self._running:
            if time.time() - start_wait > timeout:
                raise RuntimeError("Timeout waiting for event loop to start")
            time.sleep(0.001)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the asyncio event loop.

        This should be called after the CSP graph has finished to clean up
        the background thread.

        Args:
            timeout: Maximum time to wait for the thread to stop.
        """
        if not self._running:
            return

        self._running = False
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass  # Loop already closed
        if self._thread:
            self._thread.join(timeout=timeout)
        self._loop = None
        self._thread = None

    def _run_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def push(self, value: Any) -> bool:
        """
        Push a value to the CSP graph through the adapter.

        This is thread-safe and can be called from any thread, including
        asyncio callbacks.

        Args:
            value: The value to push. Must be compatible with the adapter_type.

        Returns:
            True if the push was successful, False if the adapter is not
            yet bound to a running graph.
        """
        return self._adapter.push_tick(value)

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> None:
        """
        Schedule a callback to run as soon as possible.

        The callback will execute in the asyncio thread but can push
        data to the CSP graph.

        Args:
            callback: The callback to execute.
            *args: Arguments to pass to the callback.

        Raises:
            RuntimeError: If the bridge has not been started.
        """
        if self._loop is None:
            raise RuntimeError("Bridge not started - call start() first")

        def wrapped():
            try:
                callback(*args)
            except Exception as e:
                import sys

                print(f"Error in callback: {e}", file=sys.stderr)

        self._loop.call_soon_threadsafe(wrapped)

    def call_later(self, delay: float, callback: Callable[..., Any], *args: Any) -> "asyncio.TimerHandle":
        """
        Schedule a callback after delay seconds.

        The callback will execute in the asyncio thread after the specified
        delay. This uses wall-clock time, not CSP engine time.

        Args:
            delay: Seconds to wait before calling. Must be non-negative.
            callback: The callback to execute.
            *args: Arguments to pass to the callback.

        Returns:
            A TimerHandle that can be used to cancel the callback.

        Raises:
            RuntimeError: If the bridge has not been started.
            ValueError: If delay is negative.
        """
        if self._loop is None:
            raise RuntimeError("Bridge not started - call start() first")
        if delay < 0:
            raise ValueError("delay must be non-negative")

        # Use call_soon_threadsafe to schedule the call_later
        handle_container = []

        def schedule():
            handle = self._loop.call_later(delay, callback, *args)
            handle_container.append(handle)

        self._loop.call_soon_threadsafe(schedule)

        # Return a wrapper that will eventually contain the handle
        # Note: The actual handle may not be available immediately
        return _DeferredHandle(handle_container, self._loop)

    def call_at(self, when: datetime, callback: Callable[..., Any], *args: Any) -> "asyncio.TimerHandle":
        """
        Schedule a callback at a specific datetime.

        This calculates the delay from the current wall-clock time to the
        target time and schedules accordingly. If the target time is in
        the past, the callback is scheduled immediately.

        Args:
            when: The datetime to execute the callback.
            callback: The callback to execute.
            *args: Arguments to pass to the callback.

        Returns:
            A TimerHandle that can be used to cancel the callback.

        Raises:
            RuntimeError: If the bridge has not been started.
        """
        now = datetime.utcnow()
        delay = max(0.0, (when - now).total_seconds())
        return self.call_later(delay, callback, *args)

    def call_at_offset(self, offset: timedelta, callback: Callable[..., Any], *args: Any) -> "asyncio.TimerHandle":
        """
        Schedule a callback at a specific offset from the start time.

        This is useful for scheduling callbacks aligned with CSP's engine
        start time. The offset is from the start_time provided to start().

        Args:
            offset: Time offset from start_time.
            callback: The callback to execute.
            *args: Arguments to pass to the callback.

        Returns:
            A TimerHandle that can be used to cancel the callback.

        Raises:
            RuntimeError: If the bridge has not been started.
        """
        if self._start_time is None:
            raise RuntimeError("Bridge not started - call start() first")

        target_time = self._start_time + offset
        return self.call_at(target_time, callback, *args)

    def run_coroutine(self, coro: Coroutine[Any, Any, _T]) -> "asyncio.Future[_T]":
        """
        Run an asyncio coroutine in the bridge's event loop.

        The coroutine runs in the background thread and can push data
        to the CSP graph.

        Args:
            coro: The coroutine to run.

        Returns:
            A Future representing the coroutine's result.

        Raises:
            RuntimeError: If the bridge has not been started.

        Example:
            >>> async def fetch_and_push():
            ...     data = await fetch_data()
            ...     bridge.push(data)
            ...
            >>> future = bridge.run_coroutine(fetch_and_push())
            >>> result = future.result()  # Wait for completion
        """
        if self._loop is None:
            raise RuntimeError("Bridge not started - call start() first")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def time(self) -> float:
        """
        Get current time in seconds since epoch.

        This uses wall clock time, similar to asyncio's loop.time().

        Returns:
            Current time in seconds since epoch.
        """
        return time.time()

    def elapsed_since_start(self) -> timedelta:
        """
        Get time elapsed since the bridge started.

        Returns:
            Time elapsed since start() was called.
        """
        if self._start_time is None:
            return timedelta(0)
        return datetime.utcnow() - self._start_time

    def wait_for_adapter(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the adapter to be bound to a running graph.

        This is useful to ensure the CSP graph has started and the adapter
        is ready to receive data.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if the adapter is ready, False if timeout occurred.
        """
        self._adapter.wait_for_start(timeout)
        return self._adapter.started()


class _DeferredHandle:
    """
    A handle wrapper that may not have its underlying handle immediately.

    This is used because call_later is scheduled via call_soon_threadsafe,
    so the actual TimerHandle isn't available until the event loop processes it.
    """

    def __init__(self, container: List, loop: asyncio.AbstractEventLoop):
        self._container = container
        self._loop = loop
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel the callback."""
        self._cancelled = True
        if self._container:
            self._container[0].cancel()
        else:
            # Schedule the cancel for when the handle is available
            def do_cancel():
                if self._container:
                    self._container[0].cancel()

            try:
                self._loop.call_soon_threadsafe(do_cancel)
            except RuntimeError:
                pass

    def cancelled(self) -> bool:
        """Return True if the callback was cancelled."""
        if self._cancelled:
            return True
        if self._container:
            return self._container[0].cancelled()
        return False


class BidirectionalBridge(AsyncioBridge):
    """
    Bridge supporting bidirectional communication between asyncio and CSP.

    This extends AsyncioBridge to allow not only pushing data to CSP,
    but also receiving events from CSP nodes.

    Example:
        >>> bridge = BidirectionalBridge(str)
        >>>
        >>> # Register callback to receive from CSP
        >>> bridge.on_event(lambda data: print(f"Received: {data}"))
        >>>
        >>> @csp.node
        >>> def my_node(data: ts[str], bridge_ref: object) -> ts[str]:
        ...     if csp.ticked(data):
        ...         # Emit back to asyncio
        ...         bridge_ref.emit({"response": data})
        ...         return data
    """

    def __init__(self, adapter_type: type = object, name: str = "bidi_bridge"):
        super().__init__(adapter_type, name)
        self._event_callbacks: List[Callable[[Any], None]] = []
        self._callback_lock = threading.Lock()

    def on_event(self, callback: Callable[[Any], None]) -> None:
        """
        Register a callback to receive events from CSP.

        The callback will be invoked in the asyncio thread when CSP
        nodes call emit().

        Args:
            callback: Function to call with each event.
        """
        with self._callback_lock:
            self._event_callbacks.append(callback)

    def off_event(self, callback: Callable[[Any], None]) -> bool:
        """
        Unregister an event callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed.
        """
        with self._callback_lock:
            try:
                self._event_callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def emit(self, value: Any) -> None:
        """
        Emit an event from CSP to asyncio callbacks.

        This should be called from within CSP nodes to send data back
        to the asyncio side.

        Args:
            value: The value to emit to registered callbacks.
        """
        with self._callback_lock:
            callbacks = list(self._event_callbacks)

        for callback in callbacks:
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(callback, value)
                except RuntimeError:
                    pass  # Loop closed


# Export for convenience
__all__ = ["AsyncioBridge", "BidirectionalBridge"]
