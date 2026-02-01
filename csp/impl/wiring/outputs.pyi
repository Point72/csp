"""Type stubs for csp outputs container."""

from typing import Any, Dict, Iterator, Tuple

class OutputsContainer(Dict[str, Any]):
    """
    Container for named outputs from a csp graph or node.

    This behaves like a dictionary mapping output names to their values.
    """

    def __init__(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def _items(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over (name, value) pairs."""
        ...

    def _values(self) -> Iterator[Any]:
        """Iterate over values."""
        ...

    def _get(self, item: str, dflt: Any = ...) -> Any:
        """Get a value by key with optional default."""
        ...
