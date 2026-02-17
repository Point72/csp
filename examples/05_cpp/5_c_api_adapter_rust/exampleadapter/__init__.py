"""
Example Rust adapter wiring for CSP.

This module demonstrates how to wire Rust adapters (implemented via the C API
and PyO3) into the CSP Python layer. The C API provides a stable ABI that
Rust code can implement using FFI.

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

from .exampleadapter import (
    _example_adapter_manager,
    _example_input_adapter,
    _example_output_adapter,
)

__all__ = [
    "ExampleAdapterManager",
    "example_input",
    "example_output",
]


T = TypeVar("T")


# =============================================================================
# Adapter Manager Pattern
# =============================================================================


class ExampleAdapterManager:
    """
    Example adapter manager using the Rust + C API.

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
        It creates a Rust/C API adapter manager and bridges it to CSP format.

        Args:
            engine: The PyEngine instance
            memo: Memoization dict for deduplication

        Returns:
            A capsule containing the AdapterManagerExtern*
        """
        # Create the C API manager capsule
        c_api_capsule = _example_adapter_manager(engine, self._properties)

        # Bridge to CSP format
        return _cspimpl._c_api_adapter_manager_bridge(engine, c_api_capsule)


def _create_managed_input_adapter(mgr_capsule, engine, pytype, push_mode, scalars):
    """
    Bridge function for managed input adapter.

    scalars: (ExampleAdapterManager, typ, interval_ms, push_group)
    """
    # Extract interval_ms (find first int in scalars)
    interval_ms = 100
    for s in scalars:
        if isinstance(s, int) and not isinstance(s, bool):
            interval_ms = s
            break

    # Create the VTable capsule for this adapter
    capsule = _example_input_adapter(interval_ms=interval_ms)

    # Get push group from scalars (last element if it's a PushGroup)
    push_group_capsule = None
    if len(scalars) > 0 and hasattr(scalars[-1], "__class__") and "PushGroup" in type(scalars[-1]).__name__:
        push_group_capsule = scalars[-1]

    # Pass to CSP bridge
    return _cspimpl._c_api_push_input_adapter(mgr_capsule, engine, pytype, push_mode, (capsule, push_group_capsule))


def _create_managed_output_adapter(mgr_capsule, engine, scalars):
    """
    Bridge function for managed output adapter.

    scalars: (ExampleAdapterManager, prefix)
    """
    # Extract prefix from scalars
    prefix = scalars[1] if len(scalars) > 1 else ""
    if prefix is None:
        prefix = ""

    # Create the VTable capsule
    capsule = _example_output_adapter(prefix=prefix)

    # Pass to CSP bridge
    return _cspimpl._c_api_output_adapter(mgr_capsule, engine, (int, capsule))


# Managed adapter definitions (with adapter manager)
_managed_input_adapter_def = input_adapter_def(
    "rust_input_adapter_managed",
    _create_managed_input_adapter,
    ts["T"],
    ExampleAdapterManager,  # manager_type
    typ="T",
    interval_ms=int,
    push_group=(object, None),  # Accept push_group kwarg
)

_managed_output_adapter_def = output_adapter_def(
    "rust_output_adapter_managed",
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
    Create an input adapter that generates incrementing integers via Rust + C API.

    Args:
        ts_type: The timeseries output type
        interval_ms: Interval between generated values in milliseconds
        push_mode: How to handle multiple values in same engine cycle

    Returns:
        A timeseries of the specified type
    """
    return _standalone_input_adapter_def(typ=ts_type, interval_ms=interval_ms, push_mode=push_mode)


def example_output(x: ts["T"], prefix: str = "") -> ts["T"]:
    """
    Create an output adapter that prints values via Rust + C API.

    Args:
        x: Input timeseries to print
        prefix: Optional prefix for output messages

    Returns:
        The input timeseries (pass-through)
    """
    return _standalone_output_adapter_def(x, prefix=prefix)


def _create_standalone_input_adapter(mgr, engine, pytype, push_mode, scalars):
    """Bridge function for standalone input adapter."""
    interval_ms = scalars[1] if len(scalars) > 1 else 100
    capsule = _example_input_adapter(interval_ms)
    return _cspimpl._c_api_push_input_adapter(mgr, engine, pytype, push_mode, (capsule, None))


def _create_standalone_output_adapter(mgr, engine, scalars):
    """Bridge function for standalone output adapter."""
    prefix = scalars[0] if scalars else ""
    capsule = _example_output_adapter(prefix)
    return _cspimpl._c_api_output_adapter(mgr, engine, (int, capsule))


_standalone_input_adapter_def = input_adapter_def(
    "rust_input_adapter_standalone",
    _create_standalone_input_adapter,
    ts["T"],
    typ="T",
    interval_ms=int,
)

_standalone_output_adapter_def = output_adapter_def(
    "rust_output_adapter_standalone",
    _create_standalone_output_adapter,
    input=ts["T"],
    prefix=str,
)
