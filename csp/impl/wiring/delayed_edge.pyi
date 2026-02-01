"""Type stubs for csp delayed edge."""

from typing import Generic, TypeVar

from csp.impl.types.tstype import TsType
from csp.impl.wiring.edge import Edge

T = TypeVar("T")

class DelayedEdge(Generic[T]):
    """
    A delayed edge that can be bound after graph construction.

    Useful for implementing advanced patterns where edges need to be
    connected in a deferred manner.

    Example:
        delayed = csp.DelayedEdge(ts[int])
        # ... later ...
        delayed.bind(some_edge)
    """

    def __init__(
        self,
        tstype: TsType[T],
        default_to_null: bool = ...,
    ) -> None: ...
    def bind(self, edge: Edge) -> None:
        """Bind this delayed edge to an actual edge."""
        ...

    def is_bound(self) -> bool:
        """Check if this delayed edge has been bound."""
        ...

    @property
    def tstype(self) -> TsType[T]:
        """Get the type of this delayed edge."""
        ...
