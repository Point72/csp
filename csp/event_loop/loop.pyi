"""Type stubs for csp.event_loop module."""

import asyncio
import concurrent.futures
import contextvars
import socket
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

_T = TypeVar("_T")

class _CspHandle:
    """Handle for a scheduled callback."""

    def cancel(self) -> None: ...
    def cancelled(self) -> bool: ...

class _CspTimerHandle(_CspHandle):
    """Handle for a scheduled timer callback."""

    def when(self) -> float: ...

class CspEventLoop(asyncio.AbstractEventLoop):
    """An asyncio-compatible event loop backed by CSP's scheduler."""

    def __init__(self, realtime: bool = True) -> None: ...

    # Running and stopping
    def run_forever(self) -> None: ...
    def run_until_complete(self, future: Union[asyncio.Future[_T], Coroutine[Any, Any, _T]]) -> _T: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def is_closed(self) -> bool: ...
    def close(self) -> None: ...

    # Scheduling callbacks
    def call_soon(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspHandle: ...
    def call_soon_threadsafe(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspHandle: ...
    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspTimerHandle: ...
    def call_at(
        self,
        when: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[contextvars.Context] = None,
    ) -> _CspTimerHandle: ...
    def time(self) -> float: ...

    # Futures and tasks
    def create_future(self) -> asyncio.Future[Any]: ...
    def create_task(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: Optional[str] = None,
        context: Optional[contextvars.Context] = None,
    ) -> asyncio.Task[_T]: ...
    def set_task_factory(self, factory: Optional[Callable[..., Any]]) -> None: ...
    def get_task_factory(self) -> Optional[Callable[..., Any]]: ...

    # I/O methods
    def add_reader(self, fd: int, callback: Callable[..., Any], *args: Any) -> None: ...
    def remove_reader(self, fd: int) -> bool: ...
    def add_writer(self, fd: int, callback: Callable[..., Any], *args: Any) -> None: ...
    def remove_writer(self, fd: int) -> bool: ...

    # Socket methods
    async def sock_recv(self, sock: socket.socket, nbytes: int) -> bytes: ...
    async def sock_recv_into(self, sock: socket.socket, buf: bytearray) -> int: ...
    async def sock_sendall(self, sock: socket.socket, data: bytes) -> None: ...
    async def sock_connect(self, sock: socket.socket, address: Tuple[str, int]) -> None: ...
    async def sock_accept(self, sock: socket.socket) -> Tuple[socket.socket, Tuple[str, int]]: ...

    # DNS methods
    async def getaddrinfo(
        self,
        host: Optional[str],
        port: Optional[Union[str, int]],
        *,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> List[Tuple[int, int, int, str, Tuple[str, int]]]: ...
    async def getnameinfo(self, sockaddr: Tuple[str, int], flags: int = 0) -> Tuple[str, str]: ...

    # Executor methods
    def run_in_executor(
        self,
        executor: Optional[concurrent.futures.Executor],
        func: Callable[..., _T],
        *args: Any,
    ) -> asyncio.Future[_T]: ...
    def set_default_executor(self, executor: Optional[concurrent.futures.Executor]) -> None: ...

    # Exception handling
    def set_exception_handler(self, handler: Optional[Callable[..., Any]]) -> None: ...
    def get_exception_handler(self) -> Optional[Callable[..., Any]]: ...
    def default_exception_handler(self, context: Dict[str, Any]) -> None: ...
    def call_exception_handler(self, context: Dict[str, Any]) -> None: ...

    # Debug mode
    def get_debug(self) -> bool: ...
    def set_debug(self, enabled: bool) -> None: ...

    # Signal handling
    def add_signal_handler(self, sig: int, callback: Callable[..., Any], *args: Any) -> None: ...
    def remove_signal_handler(self, sig: int) -> bool: ...

    # Shutdown
    async def shutdown_asyncgens(self) -> None: ...
    async def shutdown_default_executor(self, timeout: Optional[float] = None) -> None: ...

class CspEventLoopPolicy(asyncio.AbstractEventLoopPolicy):
    """Event loop policy for CSP-backed asyncio."""

    def __init__(self) -> None: ...
    def get_event_loop(self) -> CspEventLoop: ...
    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None: ...
    def new_event_loop(self) -> CspEventLoop: ...

# Alias
EventLoopPolicy = CspEventLoopPolicy

def new_event_loop() -> CspEventLoop:
    """Create and return a new CSP event loop."""
    ...

def run(
    main: Coroutine[Any, Any, _T],
    *,
    loop_factory: Optional[Callable[[], CspEventLoop]] = None,
    debug: Optional[bool] = None,
) -> _T:
    """Run a coroutine using the CSP event loop."""
    ...
