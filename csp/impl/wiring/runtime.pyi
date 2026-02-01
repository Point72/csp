"""Type stubs for csp runtime functions."""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from csp.impl.wiring.edge import Edge

MAX_END_TIME: datetime

class GraphRunInfo:
    """Information about the current graph run."""

    def __init__(
        self,
        starttime: Optional[datetime],
        endtime: Optional[datetime],
        realtime: bool,
    ) -> None: ...
    @property
    def starttime(self) -> Optional[datetime]: ...
    @property
    def endtime(self) -> Optional[datetime]: ...
    @property
    def is_realtime(self) -> bool: ...
    @classmethod
    def get_cur_run_times_info(
        cls,
        raise_if_missing: bool = ...,
    ) -> Optional["GraphRunInfo"]: ...
    def __enter__(self) -> "GraphRunInfo": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

def build_graph(
    f: Callable[..., Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Optional[Union[datetime, timedelta]] = ...,
    realtime: bool = ...,
    **kwargs: Any,
) -> Any:
    """
    Build a csp graph without running it.

    Args:
        f: The graph function to build
        starttime: Optional start time
        endtime: Optional end time
        realtime: Whether to configure for realtime mode

    Returns:
        A Context object representing the built graph
    """
    ...

def run(
    g: Union[Callable[..., Any], Edge, Any],
    *args: Any,
    starttime: Optional[datetime] = ...,
    endtime: Union[datetime, timedelta] = ...,
    queue_wait_time: Optional[timedelta] = ...,
    realtime: bool = ...,
    output_numpy: bool = ...,
    **kwargs: Any,
) -> Dict[str, List[Tuple[datetime, Any]]]:
    """
    Run a csp graph.

    Args:
        g: The graph function, Edge, or pre-built Context to run
        starttime: Start time for the simulation (required for historical runs)
        endtime: End time for the simulation (datetime or timedelta from starttime)
        queue_wait_time: Time to wait for realtime events
        realtime: Whether to run in realtime mode
        output_numpy: Whether to return outputs as numpy arrays

    Returns:
        Dictionary mapping output names to list of (timestamp, value) tuples.
        If output_numpy is True, values may be numpy arrays.

    Example:
        @csp.graph
        def my_graph() -> ts[int]:
            return csp.const(42)

        results = csp.run(
            my_graph,
            starttime=datetime(2024, 1, 1),
            endtime=timedelta(seconds=10),
        )
    """
    ...
