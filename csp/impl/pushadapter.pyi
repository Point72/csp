"""Type stubs for csp PushInputAdapter base class."""

from datetime import datetime
from typing import Any

class PushGroup:
    """Group for batching push operations."""
    def __init__(self) -> None: ...

class PushBatch:
    """Context manager for batching push operations."""
    def __init__(self, push_group: PushGroup) -> None: ...
    def __enter__(self) -> "PushBatch": ...
    def __exit__(self, *args: Any) -> None: ...

class PushInputAdapter:
    """
    Base class for push input adapters.

    Push adapters receive data asynchronously from external sources
    and push it into the csp graph.

    Example:
        class MyPushAdapter(PushInputAdapter):
            def __init__(self, source):
                self.source = source

            def start(self, starttime, endtime):
                self.source.subscribe(self._on_data)

            def stop(self):
                self.source.unsubscribe()

            def _on_data(self, value):
                self.push_tick(value)
    """

    def start(self, starttime: datetime, endtime: datetime) -> None:
        """Called when the adapter should start receiving data."""
        ...

    def stop(self) -> None:
        """Called when the adapter should stop receiving data."""
        ...

    def push_tick(self, value: Any) -> None:
        """
        Push a value into the graph at the current time.

        This method is inherited from the C++ base class.
        """
        ...
