"""Type stubs for csp constants."""

from typing import Any

class _UnsetType:
    """Sentinel type for unset values."""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

UNSET: _UnsetType
"""Sentinel value indicating an unset/missing value."""

REMOVE_DYNAMIC_KEY: Any
"""Sentinel value for removing a key from a dynamic basket."""
