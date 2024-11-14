from collections import deque, namedtuple
from io import BytesIO
from typing import Dict, Literal

from csp.impl.wiring.runtime import build_graph

_KIND = Literal["output", "input", ""]
_NODE = namedtuple("NODE", ["name", "label", "kind"])
_EDGE = namedtuple("EDGE", ["start", "end"])

_GRAPHVIZ_COLORMAP: Dict[_KIND, str] = {"output": "red", "input": "cadetblue1", "": "white"}

_GRAPHVIZ_SHAPEMAP: Dict[_KIND, str] = {"output": "rarrow", "input": "rarrow", "": "box"}

_DAGRED3_COLORMAP: Dict[_KIND, str] = {
    "output": "red",
    "input": "#98f5ff",
    "": "lightgrey",
}
_DAGRED3_SHAPEMAP: Dict[_KIND, str] = {"output": "diamond", "input": "diamond", "": "rect"}

_NOTEBOOK_KIND = Literal["", "terminal", "notebook"]

__all__ = (
    "generate_graph",
    "show_graph_pil",
    "show_graph_graphviz",
    "show_graph_widget",
    "show_graph",
)


def _notebook_kind() -> _NOTEBOOK_KIND:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return "notebook"
        elif shell == "TerminalInteractiveShell":
            return "terminal"
        else:
            return ""
    except ImportError:
        return ""
    except NameError:
        return ""


def _build_graph_for_viz(graph_func, *args, **kwargs):
    graph = build_graph(graph_func, *args, **kwargs)

    rootnames = set()
    q = deque()
    for nodedef in graph.roots:
        q.append(nodedef)
        rootnames.add(str(id(nodedef)))

    nodes = []
    edges = []
    visited = set()

    while q:
        nodedef = q.popleft()
        name = str(id(nodedef))
        visited.add(nodedef)
        if name in rootnames:  # output node
            kind = "output"
        elif not sum(1 for _ in nodedef.ts_inputs()):  # input node
            kind = "input"
        else:
            kind = ""

        label = nodedef.__name__ if hasattr(nodedef, "__name__") else type(nodedef).__name__
        nodes.append(_NODE(name=name, label=label, kind=kind))

        for input in nodedef.ts_inputs():
            if input[1].nodedef not in visited:
                q.append(input[1].nodedef)
            edges.append(_EDGE(start=str(id(input[1].nodedef)), end=name))
    return nodes, edges


def _build_graphviz_graph(graph_func, *args, **kwargs):
    from graphviz import Digraph

    nodes, edges = _build_graph_for_viz(graph_func, *args, **kwargs)

    digraph = Digraph(strict=True)
    digraph.attr(rankdir="LR", size="150,150")

    for node in nodes:
        digraph.node(
            node.name,
            node.label,
            style="filled",
            fillcolor=_GRAPHVIZ_COLORMAP[node.kind],
            shape=_GRAPHVIZ_SHAPEMAP[node.kind],
        )
    for edge in edges:
        digraph.edge(edge.start, edge.end)

    return digraph


def _graphviz_to_buffer(digraph, image_format="png") -> BytesIO:
    from graphviz import ExecutableNotFound

    digraph.format = image_format
    buffer = BytesIO()

    try:
        buffer.write(digraph.pipe())
        buffer.seek(0)
        return buffer
    except ExecutableNotFound as exc:
        raise ModuleNotFoundError(
            "Must install graphviz and have `dot` available on your PATH. See https://graphviz.org for installation instructions"
        ) from exc


def generate_graph(graph_func, *args, image_format="png", **kwargs):
    """Generate a BytesIO image representation of the given graph"""
    digraph = _build_graphviz_graph(graph_func, *args, **kwargs)
    return _graphviz_to_buffer(digraph=digraph, image_format=image_format)


def show_graph_pil(graph_func, *args, **kwargs):
    buffer = generate_graph(graph_func, *args, image_format="png", **kwargs)
    try:
        from PIL import Image
    except ImportError:
        raise ModuleNotFoundError(
            "csp requires `pillow` to display images. Install `pillow` with your python package manager, or pass `graph_filename` to generate a file output."
        )
    image = Image.open(buffer)
    image.show()


def show_graph_graphviz(graph_func, *args, graph_filename=None, **kwargs):
    # extract the format of the image
    image_format = graph_filename.split(".")[-1] if graph_filename else "png"

    # Generate graph with graphviz
    digraph = _build_graphviz_graph(graph_func, *args, **kwargs)

    if graph_filename:
        # output to file
        buffer = _graphviz_to_buffer(digraph=digraph, image_format=image_format)
        with open(graph_filename, "wb") as f:
            f.write(buffer.read())
    return digraph


def show_graph_widget(graph_func, *args, **kwargs):
    try:
        import ipydagred3
    except ImportError:
        raise ModuleNotFoundError(
            "csp requires `ipydagred3` to display graph widget. Install `ipydagred3` with your python package manager, or pass `graph_filename` to generate a file output."
        )

    nodes, edges = _build_graph_for_viz(graph_func=graph_func, *args, **kwargs)

    graph = ipydagred3.Graph(directed=True, attrs=dict(rankdir="LR"))

    for node in nodes:
        graph.addNode(
            ipydagred3.Node(
                name=node.name,
                label=node.label,
                shape=_DAGRED3_SHAPEMAP[node.kind],
                style=f"fill: {_DAGRED3_COLORMAP[node.kind]}",
            )
        )
    for edge in edges:
        graph.addEdge(edge.start, edge.end)
    return ipydagred3.DagreD3Widget(graph=graph)


def show_graph(graph_func, *args, graph_filename=None, **kwargs):
    # check if we're in jupyter
    if _notebook_kind() == "notebook":
        _HAVE_INTERACTIVE = True
    else:
        _HAVE_INTERACTIVE = False

    if graph_filename == "widget" and not _HAVE_INTERACTIVE:
        # widget only works in Jupyter for now
        raise RuntimeError("Interactive graph viewer only works in Jupyter.")
    elif graph_filename == "widget":
        # render with ipydagred3
        return show_graph_widget(graph_func, *args, **kwargs)
    elif graph_filename in ("", None) and not _HAVE_INTERACTIVE:
        # render with pillow
        return show_graph_pil(graph_func, *args, **kwargs)
    # render with graphviz
    return show_graph_graphviz(graph_func, *args, graph_filename=graph_filename, **kwargs)
