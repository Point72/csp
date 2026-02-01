"""Type stubs for csp node decorator."""

from typing import Any, Callable, Optional, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])

class NodeDefMeta(type):
    """Metaclass for node definitions."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def using(cls, name: Optional[str] = ..., **forced_tvars: Any) -> Callable[..., Any]: ...

class NodeDef:
    """Represents an instance of a wiring-time node."""

    def __init__(
        self,
        inputs: Any,
        scalars: Any,
        tvars: Any,
        impl: Any,
        pre_create_hook: Any,
    ) -> None: ...

@overload
def node(func: _F) -> _F: ...
@overload
def node(
    func: None = ...,
    *,
    memoize: bool = ...,
    force_memoize: bool = ...,
    debug_print: bool = ...,
    cppimpl: Optional[Any] = ...,
    name: Optional[str] = ...,
) -> Callable[[_F], _F]:
    """
    Decorator to define a csp node.

    A node is a stateful processing unit that transforms time series inputs
    into time series outputs.

    Args:
        func: The function to decorate
        memoize: Whether to memoize the node (default True)
        force_memoize: Force memoization even if csp.memoize(False) was called
        debug_print: Print the processed function for debugging
        cppimpl: C++ implementation to use
        name: Custom name for the node type
    """
    ...
