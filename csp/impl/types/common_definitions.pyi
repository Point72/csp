"""Type stubs for csp common definitions."""

from enum import IntEnum
from typing import Any, Generic, List, Optional, TypeVar, Union

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

class OutputTypeError(TypeError):
    """Base exception for output type errors."""

    ...

class OutputMixedNamedAndUnnamedError(OutputTypeError):
    """Raised when outputs mix named and unnamed styles."""

    ...

class OutputBasketNotABasket(OutputTypeError):
    """Raised when OutputBasket is given a non-basket type."""

    ...

class OutputBasketMixedShapeAndShapeOf(OutputTypeError):
    """Raised when both shape and shape_of are specified."""

    ...

class OutputBasketWrongShapeType(OutputTypeError):
    """Raised when shape/shape_of has wrong type."""

    ...

class Outputs(Generic[T1, T2]):
    """
    Define output types for a node or graph.

    Single unnamed output:
        def my_node(...) -> ts[int]: ...

    Single named output (same as unnamed):
        def my_node(...) -> Outputs(ts[int]): ...

    Multiple named outputs:
        def my_node(...) -> Outputs(x=ts[int], y=ts[float]): ...

    Basket outputs:
        def my_node(...) -> Outputs(basket=OutputBasket(Dict[str, ts[int]], shape=["a", "b"])): ...
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Outputs": ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class OutputBasket:
    """
    Define a basket output with explicit shape.

    Usage:
        OutputBasket(Dict[str, ts[int]], shape=["key1", "key2"])
        OutputBasket(List[ts[float]], shape=5)
        OutputBasket(Dict[str, ts[int]], shape_of="input_basket")

    Args:
        typ: The basket type (Dict[K, ts[V]] or List[ts[V]])
        shape: Static shape (list/tuple of keys for dict, int for list)
        shape_of: Name of input argument to copy shape from
    """

    typ: type
    shape: Union[List[Any], int, str, None]
    shape_func: Optional[str]

    def __new__(
        cls,
        typ: type,
        shape: Optional[Union[List[Any], int, str]] = ...,
        shape_of: Optional[str] = ...,
    ) -> "OutputBasket": ...

class PushMode(IntEnum):
    """
    Push mode for input adapters.

    Determines how multiple values pushed at the same timestamp are handled.
    """

    LAST_VALUE: int
    """Only keep the last value pushed at each timestamp."""

    NON_COLLAPSING: int
    """Keep all values pushed, processing them in order."""

    BURST: int
    """Collect all values at a timestamp into a list."""

class DuplicatePolicy(IntEnum):
    """Policy for handling duplicate timestamps in history lookups."""

    LAST_VALUE: int
    """Return the last value at the given timestamp."""

class ArgKind(IntEnum):
    """Kind of argument in a node/graph signature."""

    TS: int
    ALARM: int
    BASKET_TS: int
    DYNAMIC_BASKET_TS: int
    SCALAR: int
    ...

class OutputDef:
    """Definition of an output argument."""

    name: Optional[str]
    typ: type
    kind: ArgKind
    ts_idx: int
    shape: Any

    def __init__(
        self,
        name: Optional[str],
        typ: type,
        kind: ArgKind,
        ts_idx: int,
        shape: Any,
    ) -> None: ...
