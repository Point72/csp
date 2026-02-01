"""Type stubs for csp Enum."""

from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

_T = TypeVar("_T", bound="Enum")

class EnumMeta(type):
    """Metaclass for csp Enum types."""

    __metadata__: Dict[str, int]

    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        dct: Dict[str, Any],
    ) -> "EnumMeta": ...
    def __iter__(self) -> Iterator["Enum"]: ...
    @property
    def __members__(self) -> Mapping[str, "Enum"]: ...

class Enum(metaclass=EnumMeta):
    """
    Base class for csp enum types.

    Example:
        class Side(csp.Enum):
            BID = 0
            ASK = 1
    """

    name: str
    value: int
    auto: type  # enum.auto

    def __init__(self, value: Union[int, str]) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: "Enum") -> bool: ...
    def __le__(self, other: "Enum") -> bool: ...
    def __gt__(self, other: "Enum") -> bool: ...
    def __ge__(self, other: "Enum") -> bool: ...
    def __reduce__(self) -> Tuple[Type["Enum"], Tuple[int]]: ...

def DynamicEnum(
    name: str,
    values: Union[Dict[str, int], list],
    start: int = ...,
    module_name: Optional[str] = ...,
) -> Type[Enum]:
    """
    Create a dynamic enum at runtime.

    Args:
        name: Name for the enum class
        values: Dictionary of name to value mappings, or list of names
        start: Starting value for auto-numbering (default 0)
        module_name: Module name for the enum class

    Returns:
        A new Enum subclass
    """
    ...
