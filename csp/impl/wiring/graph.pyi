"""Type stubs for csp graph decorator."""

from typing import Any, Callable, Optional, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])

class GraphDefMeta(type):
    """Metaclass for graph definitions."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any: ...
    def using(cls, **forced_tvars: Any) -> Callable[..., Any]: ...

@overload
def graph(func: _F) -> _F: ...
@overload
def graph(
    func: None = ...,
    *,
    memoize: bool = ...,
    force_memoize: bool = ...,
    name: Optional[str] = ...,
    debug_print: bool = ...,
) -> Callable[[_F], _F]:
    """
    Decorator to define a csp graph.

    A graph is a composition of nodes and other graphs.

    Args:
        func: The function to decorate
        memoize: Whether to memoize the graph (default True)
        force_memoize: Force memoization even if csp.memoize(False) was called
        name: Custom name for the graph type
        debug_print: Print the processed function for debugging
    """
    ...
