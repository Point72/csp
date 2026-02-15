"""
Example C API adapter wiring for CSP.

This module demonstrates how to wire C adapters (implemented via the C API)
into the CSP Python layer. It provides:
- ExampleAdapterManager: Coordinates multiple adapters, manages lifecycle
- Example input adapter: Generates incrementing integers
- Example output adapter: Prints values to stdout
"""

from typing import TypeVar

import csp
from csp import ts
from csp.impl.wiring import input_adapter_def, output_adapter_def
from csp.lib import _exampleadapterimpl

T = TypeVar("T")


class ExampleAdapterManager:
    """
    Example adapter manager that demonstrates the C API adapter manager pattern.

    Adapter managers coordinate the lifecycle of related adapters:
    - Start: Called when the CSP graph starts
    - Stop: Called when the CSP graph stops
    - Sim time slicing: For replay/simulation mode

    Usage:
        manager = ExampleAdapterManager(prefix="[MyApp]")

        @csp.graph
        def my_graph():
            data = manager.subscribe(int, interval_ms=100)
            manager.publish(data)
    """

    def __init__(self, prefix: str = None):
        """
        Initialize the adapter manager.

        Args:
            prefix: Optional prefix for output messages
        """
        self._prefix = prefix
        self._properties = {
            "prefix": prefix or "",
        }

    def subscribe(
        self,
        ts_type: type,
        interval_ms: int = 100,
        push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING,
    ):
        """
        Create an input adapter that generates incrementing integers.

        Args:
            ts_type: The timeseries type (typically int)
            interval_ms: Interval between generated values in milliseconds
            push_mode: How to handle incoming data

        Returns:
            A CSP timeseries of the specified type
        """
        properties = {
            "interval_ms": interval_ms,
        }
        return _example_input_adapter_def(self, ts_type, properties, push_mode=push_mode)

    def publish(self, x: ts["T"]):
        """
        Create an output adapter that prints values to stdout.

        Args:
            x: The timeseries to publish
        """
        return _example_output_adapter_def(self, x)

    def __hash__(self):
        return hash(self._prefix)

    def __eq__(self, other):
        return isinstance(other, ExampleAdapterManager) and self._prefix == other._prefix

    def _create(self, engine, memo):
        """
        Create the underlying C++ adapter manager.

        This method is called by CSP when the graph is built.
        """
        return _exampleadapterimpl._example_adapter_manager(engine, self._properties)


# Adapter definitions using the adapter manager pattern
_example_input_adapter_def = input_adapter_def(
    "example_input_adapter",
    _exampleadapterimpl._example_input_adapter,
    ts["T"],
    ExampleAdapterManager,
    typ="T",
    properties=dict,
)

_example_output_adapter_def = output_adapter_def(
    "example_output_adapter",
    _exampleadapterimpl._example_output_adapter,
    ExampleAdapterManager,
    input=ts["T"],
)


# Standalone adapters (without manager) for simple use cases
def example_input(
    ts_type: type = int,
    interval_ms: int = 100,
    push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING,
) -> ts["T"]:
    """
    Create a standalone input adapter that generates incrementing integers.

    This is a simpler API when you don't need a full adapter manager.

    Args:
        ts_type: The timeseries type (typically int)
        interval_ms: Interval between generated values in milliseconds
        push_mode: How to handle incoming data

    Returns:
        A CSP timeseries of the specified type
    """
    return _standalone_input_adapter_def(ts_type, interval_ms=interval_ms, push_mode=push_mode)


def example_output(x: ts["T"], prefix: str = None):
    """
    Create a standalone output adapter that prints values to stdout.

    This is a simpler API when you don't need a full adapter manager.

    Args:
        x: The timeseries to publish
        prefix: Optional prefix for output messages
    """
    return _standalone_output_adapter_def(x, prefix=prefix)


# Standalone adapter definitions (no manager required)
_standalone_input_adapter_def = input_adapter_def(
    "example_input_adapter_standalone",
    _exampleadapterimpl._example_input_adapter,
    ts["T"],
    typ="T",
    interval_ms=int,
)

_standalone_output_adapter_def = output_adapter_def(
    "example_output_adapter_standalone",
    _exampleadapterimpl._example_output_adapter,
    input=ts["T"],
    prefix=str,
)


# ============================================================================
# Struct C API Utilities (Phase 6 - Struct Access)
# ============================================================================

# The following functions demonstrate the C Struct API pattern.
# The actual struct inspection is done in C via CspStruct.h
#
# C API for Struct Access:
# - ccsp_struct_meta_name(meta) -> const char*
# - ccsp_struct_meta_field_count(meta) -> size_t
# - ccsp_struct_meta_field_by_index(meta, index) -> CCspStructFieldHandle
# - ccsp_struct_meta_field_by_name(meta, name) -> CCspStructFieldHandle
# - ccsp_struct_field_name(field) -> const char*
# - ccsp_struct_field_type(field) -> CCspType
# - ccsp_struct_field_is_optional(field) -> int
# - ccsp_struct_get_*(s, field, &out_value) -> CCspErrorCode
# - ccsp_struct_set_*(s, field, value) -> CCspErrorCode
# - ccsp_struct_create(meta) -> CCspStructHandle
# - ccsp_struct_destroy(s)
# - ccsp_struct_copy(s) -> CCspStructHandle


def inspect_struct_type(struct_type: type) -> dict:
    """
    Inspect a csp.Struct type's fields using the C struct API.

    This demonstrates using the C Struct API to access type metadata.
    The actual implementation calls into C code via _exampleadapterimpl.

    Note: This function requires the struct type to have a _struct_meta_capsule
    attribute, which is set up for struct types that support C API access.

    Args:
        struct_type: A csp.Struct subclass

    Returns:
        A dict containing:
        - name: The struct type name
        - field_count: Number of fields
        - is_strict: Whether the struct is strict
        - fields: List of field info dicts (name, type, is_optional)

    Raises:
        TypeError: If struct_type is not a valid struct type with C API support

    Example:
        >>> import csp
        >>> class MyStruct(csp.Struct):
        ...     x: int
        ...     y: float
        ...     name: str
        ...
        >>> # Note: This requires _struct_meta_capsule to be set up
        >>> # info = inspect_struct_type(MyStruct)
    """
    return _exampleadapterimpl._example_inspect_struct_type(struct_type)


# Type mapping for CCspType values (matches CspType.h enum)
CCSP_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "BOOL",
    2: "INT8",
    3: "UINT8",
    4: "INT16",
    5: "UINT16",
    6: "INT32",
    7: "UINT32",
    8: "INT64",
    9: "UINT64",
    10: "DOUBLE",
    11: "DATETIME",
    12: "TIMEDELTA",
    13: "DATE",
    14: "TIME",
    15: "ENUM",
    16: "STRING",
    17: "STRUCT",
    18: "ARRAY",
    19: "DIALECT_GENERIC",
}


def get_type_name(ccsp_type: int) -> str:
    """
    Get the string name for a CCspType enum value.

    Args:
        ccsp_type: The integer type code from the C API

    Returns:
        The string name of the type (e.g., "INT64", "STRING")
    """
    return CCSP_TYPE_NAMES.get(ccsp_type, f"UNKNOWN({ccsp_type})")

