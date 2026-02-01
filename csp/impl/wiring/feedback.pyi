"""Type stubs for csp feedback."""

from typing import Generic, Type, TypeVar

from csp.impl.types.tstype import TsType
from csp.impl.wiring.edge import Edge

T = TypeVar("T")

class FeedbackInputDef:
    """Internal: Represents a feedback's input adapter."""
    def __init__(self, typ: Type[T]) -> None: ...

class FeedbackOutputDef(Generic[T]):
    """Internal: Represents the feedback output adapter."""
    def __init__(self, typ: Type[T]) -> None: ...
    def bind(self, x: Edge) -> None: ...
    def out(self) -> Edge: ...

class feedback(Generic[T]):
    """
    Create a feedback loop in a csp graph.

    Feedback allows creating cycles in the graph by binding an output
    back to an input that depends on it.

    Example:
        @csp.graph
        def accumulator(x: ts[int]) -> ts[int]:
            fb = csp.feedback(int)
            accum = x + csp.merge(fb.out(), csp.const(0))
            fb.bind(accum)
            return accum

    Args:
        typ: The type of values that will flow through the feedback
    """

    def __init__(self, typ: Type[T]) -> None: ...
    def bind(self, x: TsType[T]) -> None:
        """
        Bind an edge to this feedback.

        The bound edge's values will be available via out() on the next cycle.
        """
        ...

    def out(self) -> TsType[T]:
        """
        Get the output edge of this feedback.

        Returns a time series that will receive the values bound via bind().
        """
        ...
