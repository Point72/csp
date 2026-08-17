"""
Example C API adapter wiring for CSP.

This module demonstrates how to wire C adapters (implemented via the C API)
into the CSP Python layer. The C API provides a stable ABI for implementing
adapters in C or any language with C FFI (Rust, Go, Zig, etc.).

This example provides:

1. Standalone adapters (no manager):
   - example_input: Generates incrementing integers at a configurable interval
   - example_output: Prints values to stdout with optional prefix

2. Managed adapters (with adapter manager):
   - ExampleAdapterManager: Adapter manager for coordinated lifecycles
   - ExampleAdapterManager.subscribe(): Create input adapter managed by the manager
   - ExampleAdapterManager.publish(): Create output adapter managed by the manager
"""

from typing import TypeVar

import csp
from csp import ts
from csp.impl.__cspimpl import _cspimpl
from csp.impl.pushadapter import PushGroup
from csp.impl.wiring import input_adapter_def, output_adapter_def

from . import _exampleadapterimpl

T = TypeVar("T")


# =============================================================================
# Adapter Manager Pattern
# =============================================================================


class ExampleAdapterManager:
    """
    Example adapter manager using the C API.

    The adapter manager pattern allows coordinating the lifecycle of multiple
    adapters together. All adapters under the same manager share:
    - Startup/shutdown coordination
    - Push groups for batching events
    - Common configuration (like prefix)

    Usage:
        mgr = ExampleAdapterManager(prefix="[MyApp] ")

        @csp.graph
        def my_graph():
            data = mgr.subscribe(int, interval_ms=100)
            mgr.publish(data)

        csp.run(my_graph, starttime=..., endtime=...)
    """

    def __init__(self, prefix: str = ""):
        """
        Create an adapter manager.

        Args:
            prefix: Prefix string for all output adapters
        """
        self._prefix = prefix
        self._push_group = PushGroup()
        self._properties = {"prefix": prefix}

    def subscribe(
        self,
        ts_type: type = int,
        interval_ms: int = 100,
        push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING,
    ) -> ts["T"]:
        """
        Create an input adapter managed by this manager.

        Args:
            ts_type: The timeseries type (typically int)
            interval_ms: Interval between generated values in milliseconds
            push_mode: How to handle incoming data

        Returns:
            A CSP timeseries of the specified type
        """
        return _managed_input_adapter_def(
            self, ts_type, interval_ms=interval_ms, push_mode=push_mode, push_group=self._push_group
        )

    def publish(self, x: ts["T"], prefix: str = None):
        """
        Create an output adapter managed by this manager.

        Args:
            x: The timeseries to publish
            prefix: Optional additional prefix (combined with manager prefix)
        """
        effective_prefix = prefix if prefix is not None else self._prefix
        return _managed_output_adapter_def(self, x, prefix=effective_prefix)

    def _create(self, engine, memo):
        """
        Create the adapter manager implementation.

        This is called by CSP's wiring layer when the graph is built.
        It creates a C API adapter manager and bridges it to CSP format.

        Args:
            engine: The PyEngine instance
            memo: Memoization dict for deduplication

        Returns:
            A capsule containing the AdapterManagerExtern*
        """
        # Create the C API manager capsule
        c_api_capsule = _exampleadapterimpl._example_adapter_manager(engine, self._properties)

        # Bridge to CSP format
        return _cspimpl._c_api_adapter_manager_bridge(engine, c_api_capsule)


def _create_managed_input_adapter(mgr_capsule, engine, pytype, push_mode, scalars):
    """
    Bridge function for managed input adapter.

    scalars are positional, in the order the kwargs are declared by
    _managed_input_adapter_def: (typ, interval_ms, push_group). The manager is a
    separate positional argument to input_adapter_def, so it is delivered as
    mgr_capsule rather than as a scalar.
    """
    _, interval_ms, push_group = scalars

    capsule = _exampleadapterimpl._example_input_adapter(interval_ms=interval_ms)

    return _cspimpl._c_api_push_input_adapter(mgr_capsule, engine, pytype, push_mode, (capsule, push_group))


def _create_managed_output_adapter(mgr_capsule, engine, scalars):
    """
    Bridge function for managed output adapter.

    The manager is delivered as mgr_capsule rather than as a scalar, and the ts
    input is not a scalar, so scalars is just (prefix,).
    """
    (prefix,) = scalars
    if prefix is None:
        prefix = ""

    # Create the VTable capsule
    capsule = _exampleadapterimpl._example_output_adapter(prefix=prefix)

    # Pass to CSP bridge
    return _cspimpl._c_api_output_adapter(mgr_capsule, engine, (int, capsule))


# Managed adapter definitions (with adapter manager)
_managed_input_adapter_def = input_adapter_def(
    "example_input_adapter_managed",
    _create_managed_input_adapter,
    ts["T"],
    ExampleAdapterManager,  # manager_type
    typ="T",
    interval_ms=int,
    push_group=(object, None),  # Accept push_group kwarg
)

_managed_output_adapter_def = output_adapter_def(
    "example_output_adapter_managed",
    _create_managed_output_adapter,
    ExampleAdapterManager,  # manager_type
    input=ts["T"],
    prefix=str,
)


# =============================================================================
# Standalone Adapters (no manager)
# =============================================================================


def example_input(
    ts_type: type = int,
    interval_ms: int = 100,
    push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING,
) -> ts["T"]:
    """
    Create a standalone input adapter that generates incrementing integers.

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

    Args:
        x: The timeseries to publish
        prefix: Optional prefix for output messages
    """
    return _standalone_output_adapter_def(x, prefix=prefix)


def _create_standalone_input_adapter(mgr, engine, pytype, push_mode, scalars):
    """
    Bridge function for standalone input adapter.

    There is no manager, so scalars is (typ, interval_ms).
    """
    _, interval_ms = scalars

    # Create the VTable capsule
    capsule = _exampleadapterimpl._example_input_adapter(interval_ms=interval_ms)

    # Pass to CSP bridge
    return _cspimpl._c_api_push_input_adapter(mgr, engine, pytype, push_mode, (capsule, None))


def _create_standalone_output_adapter(mgr, engine, scalars):
    """
    Bridge function for standalone output adapter.

    The ts input is not a scalar, so scalars is just (prefix,).
    """
    (prefix,) = scalars
    # Convert None to empty string as the C function expects a string
    if prefix is None:
        prefix = ""

    # Create the VTable capsule
    capsule = _exampleadapterimpl._example_output_adapter(prefix=prefix)

    # Pass to CSP bridge
    return _cspimpl._c_api_output_adapter(mgr, engine, (int, capsule))


# Standalone adapter definitions (no manager required)
_standalone_input_adapter_def = input_adapter_def(
    "example_input_adapter_standalone",
    _create_standalone_input_adapter,
    ts["T"],
    typ="T",
    interval_ms=int,
)

_standalone_output_adapter_def = output_adapter_def(
    "example_output_adapter_standalone",
    _create_standalone_output_adapter,
    input=ts["T"],
    prefix=str,
)
