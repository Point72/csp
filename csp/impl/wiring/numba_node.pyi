"""Type stubs for csp numba node."""

from typing import Any, Callable, Optional, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

@overload
def numba_node(func: F) -> F:
    """
    Decorator to define a csp node that uses Numba for JIT compilation.

    Numba nodes can provide significant performance improvements for
    numerical computations.

    Example:
        @csp.numba_node
        def fast_compute(x: ts[float]) -> ts[float]:
            if csp.ticked(x):
                return x * 2.0
    """
    ...

@overload
def numba_node(
    *,
    name: Optional[str] = ...,
    memoize: bool = ...,
    force_memoize: bool = ...,
) -> Callable[[F], F]:
    """
    Decorator to define a numba node with options.
    """
    ...
