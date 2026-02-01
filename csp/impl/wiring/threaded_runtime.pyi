"""Type stubs for csp threaded runtime."""

from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Union

from csp.impl.wiring.edge import Edge

def run_on_thread(
    g: Union[Callable[..., Any], Edge, Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Union[datetime, timedelta] = ...,
    queue_wait_time: Optional[timedelta] = ...,
    realtime: bool = ...,
    auto_shutdown: bool = ...,
    daemon: bool = ...,
    **kwargs: Any,
) -> Any:
    """
    Run a csp graph on a separate thread.

    This is useful for running realtime graphs that need to continue
    processing while the main thread does other work.

    Args:
        g: The graph function, Edge, or pre-built Context to run
        starttime: Start time for the simulation
        endtime: End time for the simulation
        queue_wait_time: Time to wait for realtime events
        realtime: Whether to run in realtime mode
        output_numpy: Whether to return outputs as numpy arrays

    Returns:
        A handle to the running graph thread
    """
    ...
