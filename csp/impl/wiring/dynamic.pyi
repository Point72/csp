"""Type stubs for csp dynamic graphs."""

from typing import Any, Callable, Dict, TypeVar

from csp.impl.types.tstype import TsType

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

def dynamic(
    trigger: Dict[TsType[K], TsType[V]],
    sub_graph: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Define a dynamic sub-graph.

    Dynamic graphs allow creating and destroying sub-graphs at runtime
    based on keys appearing in a dynamic basket trigger.

    Args:
        trigger: Dynamic basket that triggers creation/destruction of sub-graphs.
                 When a new key appears, a new sub-graph is created.
                 When a key is removed, the sub-graph is destroyed.
        sub_graph: The csp.graph function to instantiate for each key
        *args: Arguments passed to sub_graph upon instantiation
        **kwargs: Keyword arguments passed to sub_graph upon instantiation

    Returns:
        Dynamic basket(s) containing the outputs from all active sub-graphs

    Example:
        @csp.graph
        def process_symbol(symbol: str, data: ts[float]) -> ts[float]:
            return data * 2

        @csp.graph
        def main(data: Dict[ts[str], ts[float]]) -> Dict[ts[str], ts[float]]:
            return csp.dynamic(data, process_symbol, csp.snapkey(), csp.attach())
    """
    ...
