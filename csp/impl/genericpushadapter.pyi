"""Type stubs for csp GenericPushAdapter."""

from typing import Generic, Optional, TypeVar

from csp.impl.types.tstype import TsType

T = TypeVar("T")

class GenericPushAdapter(Generic[T]):
    """
    A generic push adapter for pushing data into a csp graph.

    This adapter allows external code to push values into a running
    csp graph in real-time.

    Example:
        adapter = GenericPushAdapter(float, "my_adapter")

        @csp.graph
        def my_graph():
            data = adapter.out()
            csp.print("data", data)

        # In another thread:
        adapter.wait_for_start()
        adapter.push_tick(42.0)
    """

    def __init__(self, typ: type, name: Optional[str] = ...) -> None:
        """
        Create a new GenericPushAdapter.

        Args:
            typ: Type of values this adapter will push
            name: Optional name for the adapter
        """
        ...

    def out(self) -> TsType[T]:
        """
        Get the output time series for this adapter.

        Returns:
            Time series that will receive pushed values
        """
        ...

    def push_tick(self, value: T) -> bool:
        """
        Push a value into the graph.

        Args:
            value: Value to push

        Returns:
            True if the push succeeded, False if the adapter is not running
        """
        ...

    def wait_for_start(self, timeout: Optional[float] = ...) -> None:
        """
        Wait for the adapter to start.

        Blocks until the graph is running and this adapter is ready.

        Args:
            timeout: Optional timeout in seconds
        """
        ...

    def started(self) -> bool:
        """Check if the adapter has started."""
        ...

    def stopped(self) -> bool:
        """Check if the adapter has stopped."""
        ...
