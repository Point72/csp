"""Type stubs for csp PullInputAdapter base class."""

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Optional, Tuple, TypeVar

T = TypeVar("T")

class PullInputAdapter(metaclass=ABCMeta):
    """
    Base class for pull input adapters.

    Pull adapters provide data on demand, typically from files or databases.
    The engine calls next() to get the next data point.

    Example:
        class MyPullAdapter(PullInputAdapter):
            def __init__(self, data):
                self.data = iter(data)
                super().__init__()

            def next(self):
                try:
                    return next(self.data)
                except StopIteration:
                    return None
    """

    _start_time: datetime
    _end_time: datetime

    def __init__(self) -> None: ...
    def start(self, start_time: datetime, end_time: datetime) -> None:
        """
        Called when the adapter should start.

        Args:
            start_time: The simulation/engine start time
            end_time: The simulation/engine end time
        """
        ...

    def stop(self) -> None:
        """Called when the adapter should stop."""
        ...

    @abstractmethod
    def next(self) -> Optional[Tuple[datetime, T]]:
        """
        Get the next data point.

        Returns:
            None if no more data is available, or a tuple of
            (datetime, value) for the next tick.
        """
        ...
