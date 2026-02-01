"""Type stubs for csp memoization."""

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def memoize(func: F) -> F:
    """
    Decorator to memoize a function.

    The decorated function's results will be cached based on its arguments.
    """
    ...

def csp_memoized(func: F) -> F:
    """
    Decorator to memoize a csp graph object.

    This is used internally by csp to cache graph and node instances.
    """
    ...

def csp_memoized_graph_object(
    func: Callable[..., Any],
    force_memoize: bool = ...,
    function_name: str = ...,
) -> Callable[..., Any]:
    """
    Create a memoized version of a graph object factory.

    Args:
        func: The function to memoize
        force_memoize: Force memoization even for non-hashable args
        function_name: Name to use for the function
    """
    ...

def function_full_name(func: Callable[..., Any]) -> str:
    """Get the fully qualified name of a function."""
    ...
