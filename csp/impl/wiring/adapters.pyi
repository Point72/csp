"""Type stubs for csp adapter definitions."""

from typing import Any, Callable, Type, TypeVar

from csp.impl.types.tstype import TsType

T = TypeVar("T")

def input_adapter_def(
    name: str,
    impl: Any,
    output_type: TsType[T],
    **kwargs: Any,
) -> Callable[..., TsType[T]]:
    """
    Define an input adapter.

    Input adapters bring data into the csp graph from external sources.

    Args:
        name: Name of the adapter
        impl: C++ implementation class
        output_type: Type of the output time series
        **kwargs: Additional arguments defining adapter parameters

    Returns:
        A callable that creates instances of this adapter
    """
    ...

def output_adapter_def(
    name: str,
    impl: Any,
    input_type: TsType[T],
    **kwargs: Any,
) -> Callable[..., None]:
    """
    Define an output adapter.

    Output adapters send data from the csp graph to external destinations.

    Args:
        name: Name of the adapter
        impl: C++ implementation class
        input_type: Type of the input time series
        **kwargs: Additional arguments defining adapter parameters

    Returns:
        A callable that creates instances of this adapter
    """
    ...

def py_pull_adapter_def(
    name: str,
    adapter_class: Type[Any],
    output_type: TsType[T],
    **kwargs: Any,
) -> Callable[..., TsType[T]]:
    """
    Define a Python pull adapter.

    Pull adapters request data on demand (e.g., from files or databases).

    Args:
        name: Name of the adapter
        adapter_class: Python class implementing PullInputAdapter
        output_type: Type of the output time series
        **kwargs: Additional arguments defining adapter parameters

    Returns:
        A callable that creates instances of this adapter
    """
    ...

def py_push_adapter_def(
    name: str,
    adapter_class: Type[Any],
    output_type: TsType[T],
    **kwargs: Any,
) -> Callable[..., TsType[T]]:
    """
    Define a Python push adapter.

    Push adapters receive data asynchronously (e.g., from message queues).

    Args:
        name: Name of the adapter
        adapter_class: Python class implementing PushInputAdapter
        output_type: Type of the output time series
        **kwargs: Additional arguments defining adapter parameters

    Returns:
        A callable that creates instances of this adapter
    """
    ...

def add_graph_output(name: str, edge: TsType[T]) -> None:
    """
    Add a named output to the current graph.

    This allows capturing intermediate values from within a graph
    that will be included in the run() results.

    Args:
        name: Name for this output in the results
        edge: The time series to capture
    """
    ...
