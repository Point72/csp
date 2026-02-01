"""Type stubs for csp OutputAdapter base class."""

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Any

class OutputAdapter(metaclass=ABCMeta):
    """
    Base class for output adapters.

    Output adapters receive data from the csp graph and send it
    to external destinations.

    Example:
        class MyOutputAdapter(OutputAdapter):
            def __init__(self, destination):
                self.destination = destination

            def start(self):
                self.destination.connect()

            def stop(self):
                self.destination.disconnect()

            def on_tick(self, time, value):
                self.destination.write(time, value)
    """

    @abstractmethod
    def on_tick(self, time: datetime, value: Any) -> None:
        """
        Called when the adapter receives a tick.

        Args:
            time: The timestamp of the tick
            value: The value that was ticked
        """
        ...

    def start(self) -> None:
        """Called when the adapter should start."""
        ...

    def stop(self) -> None:
        """Called when the adapter should stop."""
        ...
