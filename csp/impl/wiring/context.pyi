"""Type stubs for csp wiring context."""

from datetime import datetime
from typing import Any, Optional

class Context:
    """
    Context for graph building and execution.

    Manages the state during graph construction and provides
    access to start/end times and other graph-level settings.
    """

    start_time: Optional[datetime]
    end_time: Optional[datetime]

    def __init__(
        self,
        start_time: Optional[datetime] = ...,
        end_time: Optional[datetime] = ...,
        is_global_instance: bool = ...,
    ) -> None: ...
    def __enter__(self) -> "Context": ...
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None: ...

def new_global_context(enable: bool = ...) -> Any:
    """
    Create a new global context.

    Global contexts allow caching graph objects across multiple runs.
    """
    ...

def clear_global_context() -> None:
    """
    Clear the current global context.

    This releases any cached graph objects.
    """
    ...
