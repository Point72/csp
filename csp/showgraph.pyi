"""Type stubs for csp showgraph."""

from typing import Any, Callable, Optional

def show_graph(
    graph_func: Callable[..., Any],
    *args: Any,
    graph_filename: Optional[str] = ...,
    **kwargs: Any,
) -> Any:
    """
    Visualize a csp graph.

    Generates a visual representation of the graph structure
    showing nodes and their connections.

    Args:
        graph_func: The graph function to visualize
        *args: Arguments to pass to the graph
        graph_filename: Optional filename to save the graph image
        **kwargs: Keyword arguments to pass to the graph

    Returns:
        A graphviz object that can be displayed in Jupyter notebooks

    Note:
        Requires graphviz and pillow packages to be installed.
    """
    ...
