"""Type stubs for csp PushPullInputAdapter base class."""

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

class PushPullInputAdapter:
    """
    Base class for push-pull input adapters.

    Push-pull adapters can both pull historical data and receive
    real-time pushes. This is useful for adapters that need to
    replay historical data before switching to real-time.

    Example:
        class MyPushPullAdapter(PushPullInputAdapter):
            def start(self, starttime, endtime):
                # Start receiving real-time data
                self.connect()

            def stop(self):
                self.disconnect()

            def push_tick(self, time, value):
                # Called to push data at a specific time
                super().push_tick(time, value)
    """

    def start(self, starttime: datetime, endtime: datetime) -> None:
        """Called when the adapter should start."""
        ...

    def stop(self) -> None:
        """Called when the adapter should stop."""
        ...

    def push_tick(self, time: datetime, value: Any) -> None:
        """
        Push a value into the graph at a specific time.

        Unlike PushInputAdapter, this allows specifying the timestamp.
        This method is inherited from the C++ base class.

        Args:
            time: The timestamp for this tick
            value: The value to push
        """
        ...
